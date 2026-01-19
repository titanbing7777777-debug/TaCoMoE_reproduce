

def task_definer(task_type):
    if task_type == "targets":
        return (
            "Now you are an expert in extracting the items being discussed in a conversation. "
            "Given a conversation context (reply chain), the input utterance marked with '###Input:', and the dialogue's replying structure, "
            "you need to extract the discussed items mentioned in the input utterance. Note that: "
            "1) Each discussed item must appear in the input utterance. "
            "2) You only need to extract the discussed items from the input utterance. "
            "3) The input utterance may mention multiple items. "
            "4) Always return a JSON list of lists: [[target, sentence_id], ...]. Use [] if none found.\n"
        )
    elif task_type == "aspects":
        return (
            "Now you are an expert in extracting the aspects being discussed in a conversation. "
            "Given a conversation context (reply chain), the input utterance marked with '###Input:', and the dialogue's replying structure, "
            "you need to extract the discussed aspects mentioned in the input utterance. Note that: "
            "1) Each discussed aspect must appear in the input utterance. "
            "2) You only need to extract the discussed aspects from the input utterance. "
            "3) The input utterance may mention multiple aspects. "
            "4) Always return a JSON list of lists: [[aspect, sentence_id], ...]. Use [] if none found.\n"
        )
    elif task_type == "opinions":
        return (
            "Now you are an expert in extracting the opinions and their sentiments. "
            "Given a conversation context (reply chain), the input utterance marked with '###Input:', and the dialogue's replying structure, "
            "you need to extract the discussed opinions and their sentiment polarities from the input utterance. Note that: "
            "1) Each extracted opinion must appear in the input utterance. "
            "2) Sentiment polarities must be one of: pos, neg, other. "
            "3) For each opinion, include the sentence_id of the input utterance. "
            "4) Always return a JSON list of lists: [[opinion, sentiment, sentence_id], ...]. Use [] if none found.\n"
        )
    elif task_type == "target-aspect":
        return (
            "Now you are an expert in extracting target-aspect pairs from a conversation. "
            "You are given the entire dialogue and its replying structure. "
            "Extract every target-aspect pair mentioned anywhere in the dialogue. "
            "Note that: "
            "1) Each pair must be grounded in the dialogue context. "
            "2) A target and its aspect can appear in different utterances. "
            "3) Always return a JSON list of lists: [[target, aspect], ...]. Use [] if none found.\n"
        )
    elif task_type == "target-opinion":
        return (
            "Now you are an expert in extracting target-opinion pairs from a conversation. "
            "You are given the entire dialogue and its replying structure. "
            "Extract every target-opinion pair grounded in the dialogue. "
            "Note that: "
            "1) A target and its opinion can appear in different utterances. "
            "2) Use the overall context and reply structure to connect them. "
            "3) Always return a JSON list of lists: [[target, opinion], ...]. Use [] if none found.\n"
        )
    elif task_type == "aspect-opinion":
        return (
            "Now you are an expert in extracting aspect-opinion pairs from a conversation. "
            "You are given the entire dialogue and its replying structure. "
            "Extract every aspect-opinion pair grounded in the dialogue. "
            "Note that: "
            "1) An aspect and its opinion can appear in different utterances. "
            "2) Always return a JSON list of lists: [[aspect, opinion], ...]. Use [] if none found.\n"
        )
    elif task_type == "quadruples":
        return (
            "Now you are an expert in extracting quadruples from a conversation. "
            "Given the input utterance context and replying structure, first determine if the input utterance (###Input) contains any opinion expression. "
            "If no opinion exists in the input, output 'statement-non-opinion'. "
            "If opinions exist, extract the quadruples. Note that: "
            "1) Target and Aspect can be inferred from the context or reply structure if they are implicit or omitted in the input utterance. "
            "2) The opinion word MUST be in the input utterance. "
            "3) Sentiments must be: pos, neg, or other. "
            "4) Always return a JSON object: {\"quadruples\": [{\"opinion\": string, \"sentiment\": string, \"target\": string, \"target_sentence_id\": int, \"aspect\": string, \"aspect_sentence_id\": int}, ...]}\n"
        )
    elif task_type == "no-opinion-detection":
        return (
            "Now you are an expert in detecting whether there is any opinion expressed in an input utterance. Given a conversation that contains the input utterance and its context and the corresponding replying structure, you first need to understand the replying structure and then determine if there is any opinion expressed in the input utterance. Note that: 1) You only need to focus on the input utterance. 2) Respond with a JSON-friendly string: ['Yes'] if there is any opinion expressed, otherwise ['No'].\n"
        )
    elif task_type == "reply":
        return (
            "Now you are an expert in reconstructing dialogue reply dependencies. Only the multi-turn dialogue is provided as input; no structured graph is given. "
            "First read the entire dialogue, determine each utterance's speaker tag (e.g., <root>, <u1>, <u2>, ...), and infer which utterance replies to which. "
            "Your output must be the structured textual representation of the dialogue reply dependencies, formatted exactly as Graph[name=\"dialogue-replying-structure\"]{ entity_list = [...], triple_list = [...] }. "
            "Ensure the entity_list enumerates all dialogue nodes in order and triple_list records every directed reply relation in the form (<child> -> <parent>)[relation=\"reply\"].\n"
        )
    