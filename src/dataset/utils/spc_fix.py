import os
import io

# quick and (very) dirty script to adjust SPC file length and fade length
# note: I didn't find a way to reliably identify looping SPCs so some songs may have a lot of silence at the end
# trailing silence could be removed after transcoding to flac/mp3

# to anyone who might need this code for something else:
# I thought writing this would be quick but it turned out to be a painful mess. sorry.  ¯\_(ツ)_/¯

ignore_spcs_under_length = 18    # do not make length changes to any SPCs under this length in seconds
min_spc_length = 50              # if an SPC is under this length, set it to this length in seconds
fade_length = 0                  # replaces all SPC fadeout lengths (in milliseconds) or None to leave unchanged
spc_path = "y:/spc/to_transcode" # the script will process all SPC files in this folder (or any subfolder)
verbose = True                  # print all changes made to stdout

def read_utf8_string(file, offset, length):
    if offset is not None: file.seek(offset)
    return file.read(length).decode('utf-8').rstrip('\x00')

def write_utf8_string(file, offset, length, string, null_terminate=True):
    encoded_string = string.encode('utf-8')
    if len(encoded_string) > length:
        raise ValueError("String is too long to fit in the specified length.")

    if null_terminate == True:
        encoded_string = encoded_string.ljust(length, b'\x00')

    if offset is not None: file.seek(offset)
    file.write(encoded_string)

def read_int(file, offset, length=1):
    if offset is not None: file.seek(offset)
    return int.from_bytes(file.read(length), byteorder='little')

def write_int(file, offset, length, value):
    if offset is not None: file.seek(offset)
    file.write(value.to_bytes(length, byteorder='little'))

def get_file_size(file):
    current_pos = file.tell()
    file.seek(0, io.SEEK_END)
    file_size = file.tell()
    file.seek(current_pos)
    return file_size

def find_string_offset(file, target_string):
    target_bytes = target_string.encode('utf-8')
    current_pos = file.tell()
    content = file.read()
    offset = content.find(target_bytes)
    file.seek(current_pos)
    return offset if offset != -1 else None

def find_byte_offset(file, target_byte):
    target_byte = target_byte.to_bytes(1, byteorder='little')
    current_pos = file.tell()
    content = file.read()
    offset = content.find(target_byte)
    file.seek(current_pos)
    return offset if offset != -1 else None

def spc_fix(file_path, ignore_spcs_under_length, min_spc_length, new_fade_length, verbose):

    spc_header = "SNES-SPC700 Sound File Data"
    extended_id666_header = "xid6"
    apev2_header = "APETAGEX"

    updated_fade_length = False
    updated_length = False
    updated_extended_fade_length = False
    updated_apev2_length = False
    updated_apev2_fade_length = False

    with open(file_path, 'r+b') as file:

        file_header = read_utf8_string(file, 0, len(spc_header))
        if file_header != spc_header:
            raise ValueError(f"Incorrect SPC file header: '{file_header}'")
        
        spc_length = None
        fade_length = None
        id666_file_header = read_int(file, 35, 1)
        if id666_file_header == 26 or id666_file_header == 27: # apparently this can also be 27? ¯\_(ツ)_/¯
            
            # some sick bastard decided to mix binary and text fields with no way to tell the difference
            # this is the best I could come up with since text id666 artist field comes 1 byte after
            # the binary artist field. unless the fade length is extremely long this byte would be 0
            # for a text id666, and therefore the string length would be 0
            # hopefully there are no id666 text format SPCs with a 0 length artist name field
            #artist_name = read_utf8_string(file, 176, 32)
            id666_binary = read_int(file, 176, 1) != 0

            # an additional sanity check for binary id666 data
            # hopefully no songs longer than an hour or with fade lengths > 30s in binary format ¯\_(ツ)_/¯
            if read_int(file, 169, 3) > 3600 or read_int(file, 172, 4) > 30000:
                id666_binary = False

            # one last check to see if the spc_length is parsible as a txt int
            try:
                txt_length = read_utf8_string(file, 169, 3)
                if not txt_length.isdigit():
                    id666_binary = True
                int(txt_length)
            except: id666_binary = True

            # spc length is in seconds
            if id666_binary == False:
                try:
                    spc_length = int(read_utf8_string(file, 169, 3))
                    if read_int(file, 172, 4) == 0: fade_length = 0
                    else: fade_length = int(read_utf8_string(file, 172, 5))
                except:
                    id666_binary = True

            if id666_binary == True:
                spc_length = read_int(file, 169, 3)
                fade_length = read_int(file, 172, 4)

        # read extended id666 data at end of spc file for yet _another_ fade length field
        extended_fade_length = None
        extended_intro_length = None
        extended_loop_length = None
        extended_end_length = None
        file_size = get_file_size(file)
  
        try: extended_id666_file_header = read_utf8_string(file, 66048, len(extended_id666_header))
        except: extended_id666_file_header = ""

        if extended_id666_file_header != extended_id666_header:
            file.seek(0)
            extended_id666_file_header_offset = find_string_offset(file, extended_id666_header)
            try: extended_id666_file_header = read_utf8_string(file, extended_id666_file_header_offset, len(extended_id666_header))
            except: extended_id666_file_header = ""

        if extended_id666_file_header == extended_id666_header:
            # chunk size does not include header
            chunk_size = read_int(file, None, 4) // 4 * 4 # align to 4 bytes
            chunk_offset = file.tell()

            #print("Chunk size: ", chunk_size)
            while file.tell() < (chunk_offset + chunk_size):
                
                subchunk_id = read_int(file, None, 1)
                subchunk_type = read_int(file, None, 1)
                subchunk_size = read_int(file, None, 2)
                if subchunk_type == 0: # type 0 subchunks use only the size field for the data/value
                    subchunk_size = 0
                else:
                    subchunk_size = subchunk_size // 4 * 4 # align to 4 bytes
                subchunk_offset = file.tell()

                if subchunk_offset >= file_size:
                    break
                #print(f"Subchunk ID: '{subchunk_id}' Type: '{subchunk_type}' Size: '{subchunk_size}' Offset: '{subchunk_offset}'")

                if subchunk_id == 51: # id 51 is fadeout length in ticks (1/64000th of a second)
                    extended_fade_length_offset = file.tell()
                    extended_fade_length = read_int(file, None, 4) // 64 # convert from ticks to milliseconds
                elif subchunk_id == 48: # id 48 is intro length (in ticks)
                    extended_intro_length_offset = file.tell()
                    extended_intro_length = read_int(file, None, 4) // 64 # convert from ticks to milliseconds
                elif subchunk_id == 49: # id 49 is loop length (in ticks)
                    extended_loop_length_offset = file.tell()
                    extended_loop_length = read_int(file, None, 4) // 64 # convert from ticks to milliseconds
                elif subchunk_id == 50: # id 50 is end length (in ticks)
                    extended_end_length_offset = file.tell()
                    extended_end_length = read_int(file, None, 4) // 64 # convert from ticks to milliseconds

                file.seek(subchunk_offset + subchunk_size)
                assert file.tell() == (subchunk_offset + subchunk_size), f"Expected to be at {subchunk_offset + subchunk_size}, but at {file.tell()}"

            file.seek(chunk_offset + chunk_size)
            assert file.tell() == (chunk_offset + chunk_size), f"Expected to be at {chunk_offset + chunk_size}, but at {file.tell()}"
        
        # possible apev2 tag data at the end of the file, among others
        apev2_spc_length = None
        apev2_fade_length = None

        try: apev2_file_header = read_utf8_string(file, None, len(apev2_header))
        except: apev2_file_header = ""

        if apev2_file_header != apev2_header:
            file.seek(0)
            apev2_file_header_offset = find_string_offset(file, apev2_header)
            try: apev2_file_header = read_utf8_string(file, apev2_file_header_offset, len(apev2_header))
            except: apev2_file_header = ""

        if apev2_file_header == apev2_header:
            #print(f"Processing APEv2 tag data in {file_path}")
            version = read_int(file, None, 4)
            if version == 2000:
                tag_size = read_int(file, None, 4)
                item_count = read_int(file, None, 4)
                flags = read_int(file, None, 4)
                file.seek(file.tell() + 8)

                for i in range(item_count):
                    item_size = read_int(file, None, 4)
                    item_flags = read_int(file, None, 4)
                    null_offset = find_byte_offset(file, 0)
                    item_key = read_utf8_string(file, None, null_offset+1)

                    if item_key.lower() == "spc_length":
                        apev2_spc_length_offset = file.tell()
                        apev2_spc_length_len = item_size
                        if apev2_spc_length_len > 0: # apparently this can be 0? ¯\_(ツ)_/¯
                            apev2_spc_length = int(read_utf8_string(file, None, item_size)) // 1000 # convert milliseconds to seconds
                        #print(f"Found APEv2 SPC length: {apev2_spc_length}")
                    elif item_key.lower() == "spc_fade":
                        apev2_fade_length_offset = file.tell()
                        apev2_fade_length_len = item_size
                        if apev2_fade_length_len > 0: # apparently this can be 0? ¯\_(ツ)_/¯
                            apev2_fade_length = int(read_utf8_string(file, None, item_size)) # milliseconds
                        #print(f"Found APEv2 fade length: {apev2_fade_length}")
                    else:
                        file.seek(file.tell() + item_size)
                    
                    if file.tell() >= file_size: break

        if spc_length is not None and (spc_length >= ignore_spcs_under_length or spc_length == 0) and spc_length < min_spc_length:
            if id666_binary == True: write_int(file, 169, 3, int(min_spc_length))
            else: write_utf8_string(file, 169, 3, str(min_spc_length))
            updated_length = True
        
        # potential for a problem if the apev2_spc_length_len is too short for the new length ¯\_(ツ)_/¯
        if apev2_spc_length is not None and apev2_spc_length >= ignore_spcs_under_length and apev2_spc_length < min_spc_length:
            write_utf8_string(file, apev2_spc_length_offset, apev2_spc_length_len, str(min_spc_length*1000)) # convert seconds to milliseconds
            updated_apev2_length = True

        if new_fade_length is not None:
            if fade_length is not None and fade_length != new_fade_length:
                if id666_binary == True: write_int(file, 172, 4, int(new_fade_length))
                else: write_utf8_string(file, 172, 5, str(new_fade_length))
                updated_fade_length = True

            if extended_fade_length is not None and extended_fade_length != new_fade_length:
                file.seek(extended_fade_length_offset)
                write_int(file, None, 4, new_fade_length * 64) # convert from milliseconds to ticks (1/64000th of a second)
                updated_extended_fade_length = True

            # potential for a problem if the apev2_fade_length_len is too short for the new fade length ¯\_(ツ)_/¯
            if apev2_fade_length is not None and apev2_fade_length != new_fade_length:
                write_utf8_string(file, apev2_fade_length_offset, apev2_fade_length_len, str(new_fade_length))
                updated_apev2_fade_length = True

        if verbose == True:
            if updated_length == True:
                print(f"Updated SPC length {spc_length}s -> {min_spc_length}s")
            if updated_fade_length == True:
                print(f"Updated fade length {int(fade_length/1000)}s -> {int(new_fade_length/1000)}s")
            if updated_extended_fade_length == True:
                print(f"Updated extended fade length {int(extended_fade_length/1000)}s -> {int(new_fade_length/1000)}s")
            if updated_apev2_length == True:
                print(f"Updated APEv2 SPC length {apev2_spc_length}s -> {min_spc_length}s")
            if updated_apev2_fade_length == True:
                print(f"Updated APEv2 fade length {int(apev2_fade_length/1000)}s -> {int(new_fade_length/1000)}s")

    return updated_length or updated_fade_length or updated_extended_fade_length or updated_apev2_length or updated_apev2_fade_length

if __name__ == "__main__":

    #spc_fix("Y:\\spc\\to_transcode\\Romancing SaGa 3\\113 Victory!.spc", ignore_spcs_under_length, min_spc_length, fade_length, True)
    #exit()

    if input(f"This will modify all SPCs under '{spc_path}' Are you sure you want to continue? (y/n): ").lower() != 'y':
        exit()
    
    processed_count = 0
    modified_count = 0

    for root, _, files in os.walk(spc_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == ".spc":
                spc_path = os.path.join(root, file)
                if verbose == True: print(f"Processing '{spc_path}'")
                modified_count += int(spc_fix(spc_path, ignore_spcs_under_length, min_spc_length, fade_length, verbose))
                processed_count += 1

    print(f"Successfully processed {processed_count} SPC files")
    print(f"Modified {modified_count} SPC files")

    