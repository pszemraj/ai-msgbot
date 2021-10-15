import gc
import os
import pprint as pp
import time
from datetime import datetime
from os.path import join

from aitextgen import aitextgen

folder_path = join(os.getcwd(), r"gpt-neo_datasetv2/GPT-Neo 125M dataset 2")
verbose=False

if __name__ == "__main__":
    ai = aitextgen(model_folder=folder_path, to_gpu=False,
                   gradient_checkpointing=True)
    ai.train()
    print("loaded model - ", datetime.now())
    stay_in_chat = True
    p_list = []
    while stay_in_chat:
        user_query = str(input("enter your prompt here (write 'exit' to exit) -->")).lower()
        if user_query == 'exit':
            print("... exiting loop")
            stay_in_chat = False
            break
        st = time.time()
        p_list.append(user_query + "\n")
        p_list.append("\n")
        p_list.append("peter szemraj:" + "\n")
        this_prompt = " ".join(p_list)
        print("\n... generating... \n")
        this_result = ai.generate(
            n=1,
            top_k=10,
            batch_size=64,
            max_length=128,
            min_length=16,
            prompt=this_prompt,  # TODO add way to truncate input prompt
            temperature=0.75,
            top_p=0.9, do_sample=True, return_as_list=True,
        )

        try:
            this_result = str(this_result[0]).split('\n')
            if verbose: print("the type of the result is {} and length is {}".format(type(this_result),
                                                                                     len(this_result)))
            res_out = [str(ele).strip() for ele in this_result]
            p_out = [str(ele).strip() for ele in p_list]
            diff_list = list(set(res_out).difference(p_out))  # remove prior prompts for the response
            this_result = [str(msg) for msg in diff_list if (":" not in str(msg)) and "szemr" not in str(msg)]  # remove all names

            # TODO: if the first element in the list is short (say <10 chars) and a second element exists, join them

            if not isinstance(this_result, list): list(this_result)
            output = str(this_result[0]).strip()
            if len(output) < 15 and len(this_result) > 1:  output = output + str(this_result[1]).strip()

        except:
            print("... model response is too short or something. try continuing the conv")
            output = "what?"

        pp.pprint(output, indent=4)
        p_list.append(output + "\n")
        p_list.append("\n")
        # pp.pprint(this_result[3].strip(), indent=4)
        rt = round(time.time() - st, 1)
        gc.collect()

        print("took {runtime} seconds to generate. \n".format(runtime=rt))

    print("finished - ", datetime.now())
    print("A transcript of your chat is as follows: \n")
    p_list = [item.strip() for item in p_list]
    pp.pprint(p_list)
