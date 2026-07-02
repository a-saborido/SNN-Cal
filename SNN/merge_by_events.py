import os
import glob
import re
import struct

def merge_by_events(input_path, num_events, particle_name):
    """
    Merge individual .dat files in new files with 'num_events' each one.
    """

    # Carpeta de salida 
    output_dir = f"merged_data_{particle_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. Buscar archivos 
    pattern = os.path.join(f"{input_path}", f"{particle_name}_*.dat")
    files = glob.glob(pattern)
    files.sort(key=lambda f: [int(x) for x in re.findall(r'(\d+)', os.path.basename(f))])

    total_files = len(files)
    if not files:
        print(f"Warning: No files found in {pattern}")
        return

    print(f"Total files found: {total_files}")

    # 3. Procesamiento por bloques 
    counter = 1
    delimiter = b'EOE '
    event = 1
    for i in range(0, total_files, num_events):
        divided_files = files[i : i + num_events]
        output_filename = os.path.join(output_dir, f"merged_{particle_name}_{counter}.dat")
        try:
            with open(output_filename, 'wb') as outfile:
                for filepath in divided_files:
                    outfile.write(delimiter)
                    outfile.write(struct.pack('i', event))
                    event +=1
                    with open(filepath, 'rb') as infile:
                        data = infile.read()
                        outfile.write(data)
   
            counter += 1
            
        except Exception as e:
            print(f"  -> [ERROR] {output_filename}: {e}")
    print(f" {counter - 1} files created")



if __name__ == "__main__":
    full_path_to_data = r"/lhome/ext/uovi123/uovi123l/SNN-Cal/kaon_pruebas2" 
    
    events_per_file = 1000
    
    merge_by_events(full_path_to_data, events_per_file, "kaon")



