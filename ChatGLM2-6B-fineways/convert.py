import argparse
import json
import datetime
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
    )
    parser.add_argument(
        "--write_data",
    )
    parser.add_argument(
        "--dataset_name",
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w")
    with open(args.orig_data) as f:
        lines = f.readlines()
        num_id = 1
        for line in lines:
            data = json.loads(line)

            human_input = data['input']
            #print(data['api_list'])
            api_list = '('+','.join(data['api_list'])+')' 
            print(api_list)
            propmt = human_input["prompt"]
            index = propmt.find("API")
            #print(index)
            propmt = propmt[:index+3] + api_list + propmt[index+3:]
            #print(propmt)
            human_input_str = propmt+'\\n' + human_input["context"] + '\\n' + human_input["current_turn"]
            assistant = str(data["output"])
            conversations = [{"from": "human", "value": human_input_str},{"from": "assistant", "value": assistant}]
            # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name+"-"+str(num_id)
            item = {"id":uniq_id, "conversations": conversations}
            f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()
