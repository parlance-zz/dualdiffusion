import utils.config as config

import os

from gradio_client import Client

if __name__ == "__main__":

      dataset_path = config.DATASET_PATH
      samples = os.listdir(dataset_path)

      # resume from last processed sample
      results_path = os.path.join(dataset_path, "anomalies_results.txt")
      try:
            with open(results_path, "r", encoding="utf-8") as file:
                  lines = file.readlines()
            last_processed = lines[-1].split(":")[0]
            print("Last processed: ", last_processed)
      except:
            last_processed = None
      
      client = Client("https://yuangongfdu-ltu.hf.space/")

      # query NF LTU model with each sample to find anomalies
      with open(results_path, "w+", encoding="utf-8") as file:

            for sample in samples:
                  if last_processed and sample < last_processed:
                        continue
                  sample_path = os.path.join(dataset_path, sample)
                  
                  failed = True
                  while failed == True:            
                        try:
                              failed = False
                              result = client.predict(
                                    sample_path,
                                    "Is this music?",
                                    api_name="/predict"
                              )
                              print(sample, result)
                              file.write(f"{sample}: {result}\n")
                              file.flush()
                        except:
                              failed = True

      # write anomalies to a file
      """
      anomalies_path = os.path.join(dataset_path, "anomalies_filtered.txt")
      with open(anomalies_path, "w", encoding="utf-8") as out_file:
            with open(results_path, "r", encoding="utf-8") as file:
                  lines = file.readlines()
            for line in lines:
                  line = line.split(":")
                  if len(line) > 1:
                        if line[1].strip()[:3].lower() != "yes":
                              out_file.write(line[0] + ": " + line[1])
      """