

def task_definer(task_type):
    if task_type == "targets":
        return (
            "Now you are an expert in extracting the items being discussed in a conversation. "
            "Given a conversation that contains the input utterance and its context and the corresponding replying structure, "
            "you first need to understand the replying structure and then extract the discussed items in the input utterance. Note that: "
            "1) Each discussed item must appear in the input utterance. "
            "2) You only need to extract the discussed items from the input utterance. "
            "3) The input utterance may mention multiple items. "
            "4) Always return a JSON-friendly list formatted as [[target1,<u>], [target2,<u>], ...]; use [] when no targets are found.\n"
        )
    elif task_type == "aspects":
        return (
            "Now you are an expert in extracting the aspects being discussed in a conversation. "
            "Given a conversation that contains the input utterance and its context and the corresponding replying structure, "
            "you first need to understand the replying structure and then extract the discussed aspects in the input utterance. Note that: "
            "1) Each discussed aspect must appear in the input utterance. "
            "2) You only need to extract the discussed aspects from the input utterance. "
            "3) The input utterance may mention multiple aspects. "
            "4) Always return a JSON-friendly list formatted as [[aspect1,<u>], [aspect2,<u>], ...]; use [] when no aspects are found.\n"
        )
    elif task_type == "opinions":
        return (
            "Now you are an expert in extracting the opinions and their corresponding sentiments being discussed in a conversation. "
            "Given a conversation that contains the input utterance and its context as well as the replying structure, "
            "you must first understand the replying structure and then extract the discussed opinions in the input utterance. Note that: "
            "1) Each extracted opinion must appear in the input utterance. "
            "2) You only need to extract opinions expressed in the input utterance. "
            "3) The input utterance may contain multiple opinions. "
            "4) For each opinion, you must also determine its corresponding sentiment polarity (e.g., pos, neg, other). "
            "5) Always return a JSON-friendly list formatted as [(opinion1, sentiment1,<u>), (opinion2, sentiment2,<u>), ...]; use [] when no opinions are found."
        )
    elif task_type == "target-aspect":
        return (
            "Now you are an expert in extracting the target-aspect pairs being discussed in a conversation. "
            "For this task, the entire conversation (context plus all utterances) and its structured textual representation is provided"
            "You must understand the overall replying structure and then extract every target-aspect pair mentioned anywhere in the dialogue. "
            "Note that: "
            "1) Each pair must appear within the conversation (not necessarily in one utterance). "
            "2) Consider the global dialogue context when identifying targets and aspects. "
            "3) The conversation may mention multiple target-aspect pairs. "
            "4) Always return a JSON-friendly list formatted as [(target1, aspect1), (target2, aspect2), ...]; use [] when no pairs are found.\n"
        )
    elif task_type == "target-opinion":
        return (
            "Now you are an expert in extracting the target-opinion pairs being discussed in a conversation. "
            "For this task, you are given the whole dialogue instead of a single input utterance, so use the entire context and replying structure to determine the pairs. "
            "Note that: "
            "1) Each target-opinion pair must be grounded somewhere in the conversation. "
            "2) Opinions may occur in different turns than their targets, so leverage the full dialogue to pair them. "
            "3) The conversation may contain multiple target-opinion pairs. "
            "4) Always return a JSON-friendly list formatted as [[target1, opinion1], [target2, opinion2], ...]; use [] when no pairs are found.\n"
        )
    elif task_type == "aspect-opinion":
        return (
            "Now you are an expert in extracting the aspect-opinion pairs being discussed in a conversation. "
            "The model input for this task is the full conversation, so analyze the entire dialogue and its replying structure to find relevant aspect-opinion pairings. "
            "Note that: "
            "1) Each aspect-opinion pair must be supported somewhere in the dialogue. "
            "2) Aspects and their opinions can appear in different turns, so use the global context to connect them. "
            "3) The conversation may include multiple aspect-opinion pairs. "
            "4) Always return a JSON-friendly list formatted as [[aspect1, opinion1], [aspect2, opinion2], ...]; use [] when no pairs are found.\n"
        )
    elif task_type == "quadruples":
        return (
            "Now you are an expert in extracting quadruples in a conversation. "
            "Given the input utterance, its context, and the replying structure, first understand the replying structure, then check whether the input utterance contains any opinion expression. "
            "If the input utterance has no opinion words, output 'statement-non-opinion'. "
            "If there is at least one opinion word, extract quadruples in the form [[target1, aspect1, opinion1, sentiment1],[target2, aspect2, opinion2, sentiment2] ...]. "
            "Use 'pos' for positive sentiment, 'neg' for negative sentiment, and 'other' for any neutral, mixed, or uncertain cases. "
            "Note that: "
            "1) Every opinion must appear in the input utterance. "
            "2) Targets and aspects may be inferred from the context if they are omitted in the input, but keep the opinion in the input utterance. "
            "3) The input utterance may contain multiple quadruples. "
            "4) Always return a JSON-friendly list formatted as [[target1, aspect1, opinion1, sentiment1], [target2, aspect2, opinion2, sentiment2], ...] or 'statement-non-opinion' when none are found.\n"
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
    