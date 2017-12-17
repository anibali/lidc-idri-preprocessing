require('torch')

local DenseVolume = {}
DenseVolume.__index = DenseVolume

function DenseVolume.new(voxels, voxel_size)
  local self = {}
  setmetatable(self, {__index=DenseVolume})

  self.voxels = voxels
  assert(torch.isTensor(self.voxels) and self.voxels:dim() == 3,
    'expected first argument to be a 3D tensor')

  self.voxel_size = voxel_size
  if torch.type(self.voxel_size) == 'table' then
    self.voxel_size = torch.Tensor(self.voxel_size)
  end
  assert(torch.isTensor(self.voxel_size) and self.voxel_size:nElement() == 3,
    'expected second argument to be a 3-element tensor')

  return self
end

function DenseVolume:extract_plane(dest, plane_rows_dir, plane_cols_dir, plane_origin)
  local vsz = self.voxels:size():totable()

  local function v(row, col, depth)
    if row < 1 or col < 1 or depth < 1 or
      row > vsz[1] or col > vsz[2] or depth > vsz[3]
    then
      return -1
    else
      return self.voxels[{row, col, depth}]
    end
  end

  local row_length = dest:size(2)
  local r = 0
  local c = 0

  dest:apply(function()
    -- Position in voxel-space
    local p1 = plane_origin[1] + plane_rows_dir[1] * r + plane_cols_dir[1] * c
    local p2 = plane_origin[2] + plane_rows_dir[2] * r + plane_cols_dir[2] * c
    local p3 = plane_origin[3] + plane_rows_dir[3] * r + plane_cols_dir[3] * c

    -- Trilinear filtering
    local xi = math.floor(p1)
    local yi = math.floor(p2)
    local zi = math.floor(p3)
    local xx = p1 - xi
    local yy = p2 - yi
    local zz = p3 - zi

    c = c + 1
    if c >= row_length then
      r = r + 1
      c = 0
    end

    return
      v(xi    , yi    , zi    )  * (1 - xx)  * (1 - yy)  * (1 - zz)  +
      v(xi + 1, yi    , zi    )  * (xx)      * (1 - yy)  * (1 - zz)  +
      v(xi    , yi + 1, zi    )  * (1 - xx)  * (yy)      * (1 - zz)  +
      v(xi    , yi    , zi + 1)  * (1 - xx)  * (1 - yy)  * (zz)      +
      v(xi + 1, yi    , zi + 1)  * (xx)      * (1 - yy)  * (zz)      +
      v(xi    , yi + 1, zi + 1)  * (1- xx)   * (yy)      * (zz)      +
      v(xi + 1, yi + 1, zi    )  * (xx)      * (yy)      * (1 - zz)  +
      v(xi + 1, yi + 1, zi + 1)  * (xx)      * (yy)      * (zz)
  end)

  return dest
end

local function random_point_on_unit_sphere(rng)
  local theta = torch.uniform(rng, -math.pi, math.pi)
  local phi = torch.uniform(rng, -math.pi, math.pi)
  local sin_theta = torch.sin(theta)
  return torch.Tensor{
    sin_theta * torch.cos(phi),
    sin_theta * torch.sin(phi),
    torch.cos(theta)}
end

function DenseVolume.jitter(max_angle, max_offset, max_scale, rng)
  if rng == nil then
    rng = torch.Generator()
    torch.manualSeed(rng, torch.random())
  end

  -- Create small random rotation
  local u = random_point_on_unit_sphere(rng)
  assert(math.abs(torch.cmul(u, u):sum() - 1) < 0.001)
  local theta = torch.uniform(rng, -max_angle, max_angle)
  local c = torch.cos(theta)
  local s = torch.sin(theta)
  -- Build 3D rotation matrix
  local rotation = torch.Tensor{
    {c + u[1]*u[1]*(1 - c)     , u[1]*u[2]*(1 - c) - u[3]*s, u[1]*u[3]*(1 - c) + u[2]*s},
    {u[2]*u[1]*(1 - c) + u[3]*s, c + u[2]*(1 - c)          , u[2]*u[3]*(1 - c) - u[1]*s},
    {u[3]*u[1]*(1 - c) - u[2]*s, u[3]*u[2]*(1 - c) + u[1]*s, c + u[3]*u[3]*(1 - c)     }
  }
  local dist = torch.uniform(0, max_offset)
  local offset = random_point_on_unit_sphere(rng) * dist
  local scale = 1 + torch.uniform(rng, -max_scale, max_scale)

  return {rotation, offset, scale}
end

function DenseVolume:get_plane_definitions(centre, pixel_size, dest_size, transform)
  local rotation, offset, scale = unpack(transform)

  local pixel_size = pixel_size * scale

  planes = {
    coronal = {
      rows_dir = torch.Tensor{0, 0, 1},
      cols_dir = torch.Tensor{0, 1, 0},
      origin = torch.Tensor{
        centre[1],
        centre[2] - (dest_size[2] / 2) * pixel_size,
        centre[3] - (dest_size[1] / 2) * pixel_size}
    },
    sagittal = {
      rows_dir = torch.Tensor{0, 0, 1},
      cols_dir = torch.Tensor{1, 0, 0},
      origin = torch.Tensor{
        centre[1] - (dest_size[2] / 2) * pixel_size,
        centre[2],
        centre[3] - (dest_size[1] / 2) * pixel_size}
    },
    axial = {
      rows_dir = torch.Tensor{1, 0, 0},
      cols_dir = torch.Tensor{0, 1, 0},
      origin = torch.Tensor{
        centre[1] - (dest_size[1] / 2) * pixel_size,
        centre[2] - (dest_size[2] / 2) * pixel_size,
        centre[3]}
    }
  }

  for plane_type,plane in pairs(planes) do
    plane.origin:add(offset):cdiv(self.voxel_size)
    plane.rows_dir = torch.mv(rotation, plane.rows_dir):mul(pixel_size):cdiv(self.voxel_size)
    plane.cols_dir = torch.mv(rotation, plane.cols_dir):mul(pixel_size):cdiv(self.voxel_size)
  end

  return planes
end

-- dest: 3 x [rows] x [cols] tensor to store planes
-- pixel_size: desired size of a pixel in dest (mm)
-- centre: point in mm to focus on
function DenseVolume:to_planes_mm(dest, pixel_size, centre, transform)
  local transform = transform or {torch.eye(3), torch.zeros(3), 1}

  local planes = self:get_plane_definitions(centre, pixel_size, dest[1]:size(), transform)

  for i,plane_type in ipairs{'coronal', 'sagittal', 'axial'} do
    local plane = planes[plane_type]
    self:extract_plane(dest[i], plane.rows_dir, plane.cols_dir, plane.origin)
  end

  return dest
end

-- dest: 3 x [rows] x [cols] tensor to store planes
-- pixel_size: desired size of a pixel in dest (mm)
-- centre: point in voxel-space to focus on
function DenseVolume:to_planes(dest, pixel_size, centre, transform)
  -- Convert centre to mm
  local centre_mm = torch.Tensor(centre):cmul(self.voxel_size)

  return self:to_planes_mm(dest, pixel_size, centre_mm, transform)
end

return DenseVolume
