require('torch')
local cjson = require('cjson')
local pl = require('pl.import_into')()
local DenseVolume = require('DenseVolume')

torch.setdefaulttensortype('torch.FloatTensor')

-- Set manual seeds for reproducible RNG
torch.manualSeed(1234)
math.randomseed(1234)

-- Input directory
local stage1_dir = '/data/lidc/LIDC-IDRI_stage1'
-- Output directory
local stage2_dir = '/data/lidc/LIDC-IDRI_stage2'

-- Check that stage1_dir is an existing directory.
if not pl.path.isdir(stage1_dir) then
  error(string.format('Input directory not found: "%s"', stage1_dir))
end

-- The number of augmented variations we want to generate for each example.
-- It is possible to increase this number and run the script again.
local n_augmentations = 32

local examples = pl.dir.getallfiles(stage1_dir, '*/nodule_*_metadata.json')
local planes = torch.Tensor(3, 113, 113)

local function normalize_voxels(voxels)
  local min = -2048
  local max = 4096
  local deviation = (max - min) / 2
  local offset = -(deviation + min)
  voxels:add(offset):div(deviation)
  return voxels
end

-- Reads and parses entire JSON file
local function read_json_file(path)
  local file = io.open(path, 'r')
  if not file then return nil end
  local text = file:read('*all')
  file:close()
  return cjson.decode(text)
end

for i,example_file in ipairs(examples) do
  print(string.format('%4d/%4d', i, #examples))

  local out_dir = pl.path.join(
    stage1_dir,
    ({string.find(pl.path.relpath(example_file, stage1_dir), '(.*/nodule_.*)_metadata%.json')})[3]
  )

  local example_dir = pl.path.dirname(example_file)
  local scan_metadata = read_json_file(pl.path.join(example_dir, 'scan_metadata.json'))
  local metadata = read_json_file(example_file)

  if pl.path.isdir(out_dir) then
    -- Find the current number of augmentations we have stored
    local planes_files = pl.dir.getallfiles(out_dir, 'planes_*.t7')
    table.sort(planes_files)
    local cur_n_augs = 0
    if #planes_files > 0 then
      cur_n_augs = tonumber(({string.find(pl.path.basename(planes_files[#planes_files]), 'planes_(%d+).t7')})[3])
    end
    -- Increase the number of augmentations if necessary
    if n_augmentations > cur_n_augs then
      -- Load voxels from disk
      local int_storage = torch.IntStorage(pl.path.join(example_dir, 'scan.dat'), false)
      local voxels = torch.IntTensor(int_storage, 1, torch.LongStorage{scan_metadata.rows, scan_metadata.cols, scan_metadata.slices}):float()
      normalize_voxels(voxels)

      local pixel_size = 80 / planes:size(2)
      local centre = {metadata.y_pos, metadata.x_pos, metadata.slice_number}

      local dv = DenseVolume.new(voxels,
        {scan_metadata.row_spacing, scan_metadata.column_spacing, scan_metadata.slice_thickness})

      for j = (cur_n_augs + 1), n_augmentations do
        dv:to_planes(planes, pixel_size, centre, DenseVolume.jitter(math.pi / 90, 1.0, 0.02))
        torch.save(pl.path.join(out_dir, string.format('planes_%03d.t7', j)), planes)
      end
    end
  else
    -- Make output directory and copy metadata files across
    pl.dir.makepath(out_dir)
    pl.file.copy(pl.path.join(example_dir, 'scan_metadata.json'), pl.path.join(out_dir, 'scan_metadata.json'))
    pl.file.copy(example_file, pl.path.join(out_dir, 'nodule_metadata.json'))

    -- Load voxels from disk
    local int_storage = torch.IntStorage(pl.path.join(example_dir, 'scan.dat'), false)
    local tensor_size = torch.LongStorage{scan_metadata.rows, scan_metadata.cols, scan_metadata.slices}
    local voxels = torch.IntTensor(int_storage):view(tensor_size):float()
    normalize_voxels(voxels)

    local pixel_size = 80 / planes:size(2)
    local centre = {metadata.y_pos, metadata.x_pos, metadata.slice_number}

    local dv = DenseVolume.new(voxels,
      {scan_metadata.row_spacing, scan_metadata.column_spacing, scan_metadata.slice_thickness})

    -- Generate and save unaugmented planes
    dv:to_planes(planes, pixel_size, centre)
    torch.save(pl.path.join(out_dir, string.format('planes.t7', j)), planes)

    -- Generate and save augmented planes
    for j=1,n_augmentations do
      dv:to_planes(planes, pixel_size, centre, DenseVolume.jitter(math.pi / 90, 1.0, 0.02))
      torch.save(pl.path.join(out_dir, string.format('planes_%03d.t7', j)), planes)
    end
  end

  collectgarbage()
end
