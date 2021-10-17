import os
import utils

model_links = {
    "gpt335M_275ks_checkpoint": "https://www.dropbox.com/sh/7kyoo9462lykfhp/AACbtz0FpwEvD24J04n53LGca?dl=1",
    "gpt335M_325ks_checkpoint": "https://www.dropbox.com/sh/5qhujccnpr9b8ba/AABTU9V3N87iYy7qwWEDVfnsa?dl=1",
    "gpt-neo-125M_150ks_checkpoint": "https://www.dropbox.com/sh/e2hbxkzu1e4vtte/AACdUHz-J735F5Cn-KV4udlka?dl=1",
}
if __name__ == "__main__":
    folder_names = [dir for dir in os.listdir(os.getcwd()) if os.path.isdir(dir)]

    if "gpt2_325k_checkpoint" not in folder_names:
        utils.get_zip_URL(
            model_links["gpt335M_325ks_checkpoint"], extract_loc=os.getcwd()
        )
    if "gpt-neo_datasetv2" not in folder_names:
        utils.get_zip_URL(
            model_links["gpt-neo-125M_150ks_checkpoint"], extract_loc=os.getcwd()
        )
