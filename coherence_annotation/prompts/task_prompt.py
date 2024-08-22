CHERENCE_ANNOTATION_PROMPT = '''
You are provided with a multi-turn dialogue transcript between a User1 and an User2.
Your task is to analyze the coherence relationship of each pair of responses in the dialogue.
For each response, determine its coherence relationship with each of the subsequent responses in the dialogue, and assign coherence labels to these relationships.
The coherence labels you may use are: Comment, Clarification-Question, Elaboration, Acknowledgment, Continuation, Explanation, Conditional, QA, Alternation, Question-Elaboration, Result, Background, Narration, Correction, Parallel, Contrast, Topic Shift.
A relationship can have multiple labels if it serves multiple functions or has multiple characteristics.

For each pair of responses, provide the assigned coherence labels in a list format. Ensure that your labeling accurately reflects the nature and relationship of the pair of responses within the dialogue context.
Please refer to the following example for the json format of the coherence labels.

First give you an example to help you understand the task:
{demonstration}

Then give you the personas and dialogue history of the task:
[Dialogues]
{input}
[Coherence Labels]
'''

CHERENCE_ANNOTATION_SEPARATED_PROMPT = '''
You are provided two sentences from a dialogue.
Your task is to analyze the coherence relationship between the two sentences.
Determining the coherence relationship between two sentences, and assign coherence labels to these relationships.
The coherence labels you may use are: Comment, Clarification-Question, Elaboration, Acknowledgment, Continuation, Explanation, Conditional, QA, Alternation, Question-Elaboration, Result, Background, Narration, Correction, Parallel, Contrast, Topic Shift.
A relationship can have multiple labels if it serves multiple functions or has multiple characteristics.

Ensure that your labeling accurately reflects the nature and relationship between the two sentences.
Please refer to the following example for the output format of the coherence labels.

First give you an example to help you understand the task:
{demonstration}

Then give you the personas and dialogue history of the task:
[Dialogues]
{input}
[Coherence Labels]
'''

CHERENCE_ANNOTATION_SEPARATED_PROMPT_v2 = '''
You are provided with two sentences from a dialogue. Your task is to analyze the coherence relationship between these two sentences, specifically focusing on how Sentence 2 responds to or relates to Sentence 1. Assign appropriate coherence labels to these relationships.
The coherence labels you may use are: Comment, Clarification-Question, Elaboration, Acknowledgment, Continuation, Explanation, Conditional, QA, Alternation, Question-Elaboration, Result, Background, Narration, Correction, Parallel, Contrast, Topic Shift.

A relationship can have multiple labels if it serves multiple functions or has multiple characteristics. However, you should assign no more than four labels to any relationship.
If the two sentences are incoherent or do not have any relationship (difficult to assign a coherent relationship), return "None".

Ensure that your labeling accurately reflects how Sentence 2 responds to or relates to Sentence 1.
For each coherence labels separated by commas. Please refer to the following examples for the output format of the coherence labels.
Please do not provide any rationale or explanation for your labels. Only provide the labels.

First give you an example to help you understand the task:
{demonstration}

Then give you the personas and dialogue history of the task:
[Dialogues]
{input}
[Coherence Labels]
'''

CHERENCE_ANNOTATION_SEPARATED_PROMPT_v3 = '''
You are provided with two sentences from a dialogue. Your task is to analyze the coherence relationship between these two sentences, specifically focusing on how Sentence 2 responds to or relates to Sentence 1. Assign appropriate coherence labels to these relationships.
The coherence labels you may use are:
- **Comment**: Sentence 2 adds a comment to Sentence 1.
- **Clarification-Question**: Sentence 2 asks for clarification about Sentence 1.
- **Elaboration**: Sentence 2 provides more details or expands on Sentence 1.
- **Acknowledgment**: Sentence 2 acknowledges or confirms what was said in Sentence 1.
- **Continuation**: Sentence 2 continues the idea or topic introduced in Sentence 1.
- **Explanation**: Sentence 2 explains or provides a reason for Sentence 1.
- **Conditional**: Sentence 2 provides a condition related to Sentence 1.
- **QA**: Sentence 2 answers a question posed in Sentence 1.
- **Alternation**: Sentence 2 presents an alternative to Sentence 1.
- **Question-Elaboration**: Sentence 2 elaborates on a question posed in Sentence 1.
- **Result**: Sentence 2 states a result or outcome of Sentence 1.
- **Background**: Sentence 2 provides background information related to Sentence 1.
- **Narration**: Sentence 2 narrates a sequence of events related to Sentence 1.
- **Correction**: Sentence 2 corrects information in Sentence 1.
- **Parallel**: Sentence 2 presents parallel or similar information to Sentence 1.
- **Contrast**: Sentence 2 contrasts with Sentence 1.
- **Topic Shift**: Sentence 2 shifts to a different topic from Sentence 1.

A relationship can have multiple labels if it serves multiple functions or has multiple characteristics. However, you should assign no more than four labels to any relationship.
If the two sentences are incoherent or do not have any relationship (difficult to assign a coherent relationship), return "None".

Ensure that your labeling accurately reflects how Sentence 2 responds to or relates to Sentence 1.
For each coherence labels separated by commas. Please refer to the following examples for the output format of the coherence labels.
Please do not provide any rationale or explanation for your labels. Only provide the labels.

First give you an example to help you understand the task:
{demonstration}

Then give you the dialogues of the task:
[Dialogues]
{input}
[Coherence Labels]
'''

COHERENCE_ANNOTATION_PROMPT_ONE_SHOT_EXAMPLE = '''
[Dialogues]
User1:Hello what are doing today?
User2:I am good , i just got off work and tired, i have two jobs.
User1:I just got done watching a horror movie.
User2:I rather read, i've read about 20 books this year.
[Coherence Labels]
[
    {
        from: Hello what are doing today?,
        to: I am good , i just got off work and tired, i have two jobs.,
        from_index: 0,
        to_index: 1,
        labels: [QA, Explanation]
    },
    {
        from: Hello what are doing today?,
        to: I just got done watching a horror movie.,
        from_index: 0,
        to_index: 2,
        labels: [Topic Shift]
    },
    {
        from: Hello what are doing today?,
        to: I rather read, i've read about 20 books this year.,
        from_index: 0,
        to_index: 3,
        labels: [Topic Shift]
    },
    {
        from: I am good , i just got off work and tired, i have two jobs.,
        to: I just got done watching a horror movie.,
        from_index: 1,
        to_index: 2,
        labels: [Topic Shift]
    },
    {
        from: I am good , i just got off work and tired, i have two jobs.,
        to: I rather read, i've read about 20 books this year.,
        from_index: 1,
        to_index: 3,
        labels: [Continuation, Contrast]
    }
    {
        from: I just got done watching a horror movie.,
        to: I rather read, i've read about 20 books this year.,
        from_index: 2,
        to_index: 3,
        labels: [Contrast, Continuation]
    }
]
'''


COHERENCE_ANNOTATION_SEPARATED_PROMPT_FEW_SHOTS_EXAMPLE = '''
[Dialogues]
Hello what are doing today?
I am good , i just got off work and tired, i have two jobs.
[Coherence Labels]
QA, Explanation
[Dialogues]
Hello what are doing today?
I just got done watching a horror movie.
[Coherence Labels]
Topic Shift
'''

COHERENCE_ANNOTATION_SEPARATED_PROMPT_FEW_SHOTS_EXAMPLE_v2 = '''
[Dialogues]
Sentence 1: Hello what are doing today?
Sentence 2: I am good , i just got off work and tired, i have two jobs.
[Coherence Labels]
QA, Explanation
[Dialogues]
Sentence 1: Hello what are doing today?
Sentence 2: I just got done watching a horror movie.
[Coherence Labels]
Topic Shift
[Dialogues]
Sentence 1: Oh that is a good profession must make alot of money.
Sentence 2: It is a comfortable living. I like spending time with family.
[Coherence Labels]
Comment, Elaboration
'''