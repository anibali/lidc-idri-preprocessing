#!/usr/bin/env ruby

require "set"
require "csv"
require "json"
require "fileutils"
require "pathname"
require "nokogiri"
require "zlib"
require "narray"
require "dicom"
include DICOM

DICOM.logger.level = Logger::ERROR

IN_DIR = "/data/lidc/LIDC-IDRI"
OUT_DIR = "/data/lidc/LIDC-IDRI_stage1"
CORNELL_LIST_FILE = "./cornell_nodule_size_list.csv"

CASES_WITH_ISSUES = Set.new(%w[
  LIDC-IDRI-0107 LIDC-IDRI-0123 LIDC-IDRI-0146
  LIDC-IDRI-0340 LIDC-IDRI-0418 LIDC-IDRI-0566
  LIDC-IDRI-0572 LIDC-IDRI-0672 LIDC-IDRI-0979
])

def main()
  whitelist = Set.new

  cornell_list = {}
  header_row = true
  CSV.foreach(CORNELL_LIST_FILE) do |row|
    if header_row
      header_row = false
      next
    end

    patient_id = "LIDC-IDRI-#{row[0]}"
    series_number = row[1]

    (cornell_list["#{patient_id}/#{series_number}"] ||= []) << {
      patient_id: patient_id,
      series_number: series_number,
      roi:  row[2].to_i,
      volume: row[3].to_f,
      diameter: row[4].to_f,
      x_pos: row[5].to_i,
      y_pos: row[6].to_i,
      slice_number: row[7].to_i,
      nodule_ids: row[9..-1].compact
    }

    unless CASES_WITH_ISSUES.include?(patient_id)
      whitelist << "#{patient_id}/#{series_number}"
    end
  end

  case_dirs = Dir.glob("#{IN_DIR}/LIDC-IDRI-*").sort

  case_dirs.each do |case_dir|
    scan_dirs = Dir.glob("#{case_dir}/*")

    scan_dirs.each do |scan_dir|
      xml_files = Dir.glob(File.join(scan_dir, "**/*.xml")).sort
      next if xml_files.empty?

      xml_file = xml_files.first
      example_dir = File.dirname(xml_file)
      dcm_files = Dir.glob("#{example_dir}/*.dcm")

      first_dcm = DObject.read(dcm_files[0])
      series_number = first_dcm["0020,0011"].value
      patient_id = first_dcm["0010,0020"].value

      composite_key = "#{patient_id}/#{series_number}"

      # Process the scan directory if it is good
      if whitelist.include?(composite_key)
        puts composite_key

        out_scan_dir = File.join(OUT_DIR, composite_key)
        if File.directory?(out_scan_dir)
          next
        end

        xml_doc = open(xml_file, "r") { |f| Nokogiri::XML(f) }
        xml_doc.remove_namespaces!

        nodule_metadatas = []

        cornell_list[composite_key].each do |cornell_record|
          nodule_metadata = cornell_record.dup
          observations = []

          cornell_record[:nodule_ids].each do |nodule_id|
            reading_element =
              xml_doc.at_xpath("//unblindedReadNodule[noduleID = '#{nodule_id}']")

            # There are some nodule IDs in the Cornell list missing a leading 0,
            # so try prepending a zero
            if reading_element.nil?
              nodule_id.insert(0, '0')
              reading_element =
                xml_doc.at_xpath("//unblindedReadNodule[noduleID = '#{nodule_id}']")
            end

            if reading_element.nil?
              puts "Couldn't find nodule by ID: '#{nodule_id}'"
              observations << nil
              next
            end

            characteristics = reading_element.at_xpath("characteristics")
            if characteristics
              begin
                observations << {
                  # Higher value means more obvious
                  subtlety: characteristics.at_xpath("subtlety").text.to_i,
                  internal_structure: characteristics.at_xpath("internalStructure").text.to_i,
                  calcification: characteristics.at_xpath("calcification").text.to_i,
                  # "Roundness" of nodule
                  sphericity: characteristics.at_xpath("sphericity").text.to_i,
                  # Sharpness of nodule boundary
                  margin: characteristics.at_xpath("margin").text.to_i,
                  # Lobulation of nodule (cloud-like lumpiness)
                  lobulation: characteristics.at_xpath("lobulation").text.to_i,
                  # Spiculation of nodule (protruding spikes/points)
                  spiculation: characteristics.at_xpath("spiculation").text.to_i,
                  # "Solidness" of texture
                  texture: characteristics.at_xpath("texture").text.to_i,
                  # Subjective likelihood of cancer assuming 60 yo male smoker
                  malignancy: characteristics.at_xpath("malignancy").text.to_i
                }
              rescue => ex
                puts "Error reading characteristics: #{ex.message}"
                observations << nil
                next
              end
            end
          end
          nodule_metadata["observations"] = observations

          nodule_metadatas << nodule_metadata
        end

        slices = []
        cols = nil
        rows = nil

        dcm_files.each do |dcm_file|
          dcm = DObject.read(dcm_file)
          dcm_hash = dcm.to_hash
          if dcm.modality.value == "CT"
            cols ||= dcm_hash["Columns"].to_i
            rows ||= dcm_hash["Rows"].to_i

            slice = {
              slice_location: dcm_hash["Image Position (Patient)"].split("\\")[2].to_f,
              slice_number: dcm_hash["Instance Number"].to_i,
              data: dcm.narray
            }

            slices << slice
          end
        end

        # Sort slices in foot-to-head direction (we make the assumption that
        # smaller slice location means closer to foot)
        slices.sort_by! {|slice| slice[:slice_location]}

        first_slice_location = slices.first[:slice_location]
        last_slice_location = slices.last[:slice_location]

        slice_thickness = (last_slice_location - first_slice_location).abs / (slices.size - 1)

        scan_metadata = {
          cols: cols,
          rows: rows,
          slices: slices.size,
          slice_thickness: slice_thickness,
          row_spacing: first_dcm["0028,0030"].value.split("\\")[0].to_f,
          column_spacing: first_dcm["0028,0030"].value.split("\\")[1].to_f
        }

        Pathname.new(out_scan_dir).mkpath

        nodule_metadatas.each_with_index do |nodule_metadata, i|
          out_nodule_file = File.join(out_scan_dir, "nodule_%02d_metadata.json" % [i + 1])
          File.open(out_nodule_file, "w") do |f|
            f.write(JSON.pretty_generate(nodule_metadata))
          end
        end

        out_scan_file = File.join(out_scan_dir, "scan_metadata.json")
        File.open(out_scan_file, "w") do |f|
          f.write(JSON.pretty_generate(scan_metadata))
        end

        data_block = NArray.int(slices.size, rows, cols)

        # Write slices in craniocaudal direction (head-to-foot)
        slices.reverse.each_with_index do |slice, i|
          data_block[i, true, true] = slice[:data]
        end

        # Rescale values
        rescale_intercept = first_dcm["0028,1052"].value.to_f
        rescale_slope = first_dcm["0028,1053"].value.to_f
        data_block.mul!(rescale_slope)
        data_block.add!(rescale_intercept)

        out_data_file = File.join(out_scan_dir, "scan.dat")
        data_string = data_block.to_s
        # data_string = Zlib::Deflate.deflate(data_string, 6)
        File.open(out_data_file, "wb") do |f|
          f.write(data_string)
        end
      end
    end
  end
end

main()
