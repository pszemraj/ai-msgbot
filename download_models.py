import os
import utils

model_links = {
    "gpt335M_275ks_checkpoint": "https://www.dropbox.com/sh/7kyoo9462lykfhp/AACbtz0FpwEvD24J04n53LGca?dl=1",
    "gpt335M_325ks_checkpoint": "https://www.dropbox.com/sh/5qhujccnpr9b8ba/AABTU9V3N87iYy7qwWEDVfnsa?dl=1",
    "gpt-neo-125M_150ks_checkpoint": "https://www.dropbox.com/sh/e2hbxkzu1e4vtte/AACdUHz-J735F5Cn-KV4udlka?dl=1",
    "gpt2_std_gpu_774M_60ksteps":"https://www.dropbox.com/sh/2wu7nbckqo5ghga/AABMg6SUaaP103WcL2lnF2b7a?dl=1",
    "gpt2_std_gpu_774M_120ksteps":"https://www.dropbox.com/sh/f8pocv18n0bohng/AACVMXcWR9Kn_CQsZKqpF1xoa?dl=1",
    "gpt2_dailydialogue_355M_75Ksteps":"https://www.dropbox.com/sh/ahx3teywshods41/AACrGhc_Qntw6GuX7ww-3pbBa?dl=1",
}
if __name__ == "__main__":

    folder_names = [dir for dir in os.listdir(os.getcwd()) if os.path.isdir(dir)]

    if "gpt2_325k_checkpoint" not in folder_names:
        # standard GPT-2 trained in a mediocre way up to 325,000 steps on my whats app data
        utils.get_zip_URL(
            model_links["gpt335M_325ks_checkpoint"], extract_loc=os.getcwd()
        )
    if "gpt-neo_datasetv2" not in folder_names:
        # the GPT-Neo small model by EleutherAI trained on my whatsapp data (this was the first one I made.. might suck
        utils.get_zip_URL(
            model_links["gpt-neo-125M_150ks_checkpoint"], extract_loc=os.getcwd()
        )

    if "gpt2_dailydialogue_355M_75Ksteps" not in folder_names:
        # this is the generic" chatbot pretrain, that other models should be trained from. it can also be used standalone.. but untest
        utils.get_zip_URL(
            model_links["ggpt2_dailydialogue_355M_75Ksteps"], extract_loc=os.getcwd()
        )



    if "gpt2_std_gpu_774M_120ksteps" not in folder_names:
        # current PROD model in telegram
        utils.get_zip_URL(
            model_links["gpt2_std_gpu_774M_120ksteps"], extract_loc=os.getcwd()
        )
