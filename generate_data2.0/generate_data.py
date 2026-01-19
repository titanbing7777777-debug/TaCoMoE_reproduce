from pathlib import Path
import json
from utils import task_definer

def find_reply_chain(replies, current):
    chain = []
    while(current != -1):
        chain.append(current)
        current = replies[current]
    chain.reverse()
    return chain

def _normalize_target(value):
    """Ensure target field is always a string for JSONL loading."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)

task_type = ["no-opinion-detection","reply","targets","aspects","opinions","target-aspect","target-opinion","aspect-opinion","quadruples"]
modes = ["train","valid","test"]
language = ["en","zh"]


BASE = Path(__file__).resolve().parent.parent  # /Fullmoon717/TaCoMoE
dataset_root = BASE / "dataset"

def process_data(dataset_path, outdir_path,mode,language):
    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # ...
    for data_id, d in enumerate(data):
        inputs_onebyone = []
        # print(d)
        speakers = d["speakers"]
        replies = d["replies"]
        sentences = d["sentences"]
        dialogue_length = len(d["sentences"]) 

        context = "###Context:\n"
        input_sentences = []
        reply_prompt = "###Replying structure:\nGraph[name=\"dialogue-replying-structure\"]"

        entity_list = []
        triple_list = []
        word_spilid = []
        context = []
        reply_chain = []

        for i in range(dialogue_length):
            reply_chain.append(find_reply_chain(replies, i))
            if i == 0:
                context.append(f"<root>speaker{speakers[i]}:{sentences[i]}\n")
                entity_list.append('<root>')
                triple_list.append('(root -> root)[relation=\"reply\"]')
                word_spilid.append(len(sentences[i].split()))
            else:
                context.append(f"<u{i}>speaker{speakers[i]}:{sentences[i]}\n")
                entity_list.append(f'<u{i}>')
                triple_list.append(f'(<u{i}> -> <u{replies[i]}>)[relation=\"reply\"]')
                word_spilid.append(word_spilid[i-1] + len(sentences[i].split()))

        # 逐句输入的结构
        for i in range(dialogue_length):
            input_sentence = sentences[i]
            chain_context = "".join([context[j] for j in reply_chain[i]])
            input_onebyone = f"""
                {chain_context}
                ###Input:{input_sentence}
                {reply_prompt}{{
                    entity_list = {entity_list}
                    triple_list = {triple_list}
                }}
            ###Answer"""
            # print(input_onebyone) 
            inputs_onebyone.append(input_onebyone)

        # 整段输入的结构
        input_dialogue = f"""
            {"".join(context)}
            {reply_prompt}{{
                entity_list = {entity_list}
                triple_list = {triple_list}
            }}
        ###Answer"""


        targets = d["targets"]
        aspects = d["aspects"]
        opinions = d["opinions"]
        triples = d["triplets"]

        target_u = [[] for _ in range(dialogue_length)]
        aspect_u = [[] for _ in range(dialogue_length)]
        opinion_u = [[] for _ in range(dialogue_length)]

        opinion_utterance = [False for _ in range(dialogue_length)]

        for t in targets:
            for uid, ws in enumerate(word_spilid):
                if t[0] < ws:
                    target_u[uid].append((t[2],uid))
                    break
        
        for a in aspects:
            for uid, ws in enumerate(word_spilid):
                if a[0] < ws:
                    aspect_u[uid].append((a[2],uid))
                    break
        
        for o in opinions:
            for uid, ws in enumerate(word_spilid):
                if o[0] < ws:
                    opinion_u[uid].append((o[2],o[3],uid))
                    break

        for op_id, op in enumerate(opinion_u):
            if len(op) > 0:
                opinion_utterance[op_id] = True

        target_aspect = []
        target_opinion = []
        aspect_opinion = []
        quadruple = []
        quadruple_u = [[] for _ in range(dialogue_length)]

        for t in triples:
            if t[7] and t[8]:
                target_aspect.append((t[7],t[8]))
            if t[7] and t[9]:
                target_opinion.append((t[7],t[9]))
            if t[8] and t[9]:
                aspect_opinion.append((t[8],t[9]))
            if t[7] and t[8] and t[9] and t[6] != -1:
                quadruple.append((t[7],t[8],t[9],t[6],t[0],t[2],t[4]))

        res = {'pos': 'pos', 'neg': 'neg'}
        for q in quadruple:
            for uid, ws in enumerate(word_spilid):
                if q[4] < ws:
                    target_sentence_index = uid
                if q[5] < ws:
                    aspect_sentence_index = uid
                if q[6] < ws:
                    current_quadruple = {
                        "opinion":q[2],
                        "sentiment":res.get(q[3], 'other'),
                        "target":q[0],
                        "target_sentence_id":target_sentence_index,
                        "aspect":q[1],
                        "aspect_sentence_id":aspect_sentence_index
                    }
                    quadruple_u[uid].append(current_quadruple)
                    break

        for task in task_type:
            if task == "reply":
                data_dict = {
                "input":task_definer(task) + input_dialogue,
                "task_type":task,
                "target":f"{reply_prompt}{{\n    entity_list = {entity_list},\n    triple_list = {triple_list}\n}}",
                "task_dataset":f"{task}",
                "sample_id":f"{task}_{data_id}"
            }
                with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                    f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")

            # targets, aspects, opinions 需要针对每一句话
            if task == "targets":
                for uid, Input in enumerate(inputs_onebyone):
                    data_dict = {
                        "input":task_definer(task) + Input,
                        "task_type":task,
                        "target":_normalize_target(target_u[uid]),
                        "task_dataset":f"{task}",
                        "sample_id":f"{task}_{data_id}_{uid}"
                    }
                    with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                        f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            if task == "aspects":
                for uid, Input in enumerate(inputs_onebyone):
                    data_dict = {
                        "input":task_definer(task) + Input,
                        "task_type":task,
                        "target":_normalize_target(aspect_u[uid]),
                        "task_dataset":f"{task}",
                        "sample_id":f"{task}_{data_id}_{uid}"
                    }
                    with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                        f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            if task == "opinions":
                for uid, Input in enumerate(inputs_onebyone):
                    data_dict = {
                        "input":task_definer(task) + Input,
                        "task_type":task,
                        "target":_normalize_target(opinion_u[uid]),
                        "task_dataset":f"{task}",
                        "sample_id":f"{task}_{data_id}_{uid}"
                    }
                    with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                        f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")

            # pair 只需要整段对话的，不需要针对每一句
            if task == "target-aspect":
                data_dict = {
                    "input":task_definer(task) + input_dialogue,
                    "task_type":task,
                    "target":_normalize_target(target_aspect),
                    "task_dataset":f"{task}",
                    "sample_id":f"{task}_{data_id}"
                }
                with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                    f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            if task == "target-opinion":
                data_dict = {
                    "input":task_definer(task) + input_dialogue,
                    "task_type":task,
                    "target":_normalize_target(target_opinion),
                    "task_dataset":f"{task}",
                    "sample_id":f"{task}_{data_id}"
                }
                with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                    f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            if task == "aspect-opinion":
                data_dict = {
                    "input":task_definer(task) + input_dialogue,
                    "task_type":task,
                    "target":_normalize_target(aspect_opinion),
                    "task_dataset":f"{task}",
                    "sample_id":f"{task}_{data_id}"
                }
                with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                    f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")

            if task == "quadruples":
                for uid, Input in enumerate(inputs_onebyone):
                    if len(quadruple_u[uid]) == 0:
                        target_value = "statement-non-opinion"
                    else:
                        target_value = {"quadruples": quadruple_u[uid]}
                        print(target_value)
                        target_value = _normalize_target(target_value)
                    data_dict = {
                        "input":task_definer(task) + Input,
                        "task_type":task,
                        "target":target_value,
                        "task_dataset":f"{task}",
                        "sample_id":f"{task}_{data_id}_{uid}"
                    }
                    with open(outdir_path / f"{mode}.json","a",encoding="utf-8") as f:
                        f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for l in language:
        for m in modes:
            process_data(dataset_root / f"jsons_{l}/{m}.json", BASE / f"data/{l}/task/",m,l)
            print(f"完成数据集生成：{l}-{m}")