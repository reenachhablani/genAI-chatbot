# genAI-chatbot

## Objective - Developing a chatbot that interacts with users to suggest addons based on their specific requirements.

## Proposed User Interaction:
There will be an interactive chatbot in the bottom right corner of the add-on website which will prompt the user for any help they need.
When user clicks on the chat icon. It will ask for the requirement. It will suggest the top addon names, icons, description, and links relevant to their provided prompt.
Based on the user's maturity level, if they aren't satisfied with the initial suggestions, prompt them to provide more specific details about their requirement and  an iterative process will be continued as per the prompts.

# Implementation Details

Currently, I have tried to implement a prototype in which the user can input a prompt for the required add-on and the results will be generated.
## Dataset:

I have scraped the mozilla add-ons website and created a dummy dataset for the prototype. - 
The dataset includes the fields- addon name, description, link, ratings.
For Example -
{'Name': 'Tab Reloader (page auto refresh)',
 'Link': '/en-US/firefox/addon/tab-reloader/?utm_source=addons.mozilla.org&utm_medium=referral&utm_content=featured',
 'Description': 'An easy-to-use tab reloader with custom reloading time settings for individual tabs'
‘Rating’: 4.7}

This dataset has been build considering four categories from the mozilla add-on server data available on github.

## LLM:

I have explored various open-source models out of which I will like to mention GPT-J, Falcom 40B, Llama 2. 
I have currently implemented Llama-2 model and fine-tuned on the given scraped dataset using Pytorch and AWS because of various reasons - 
Llama-2 is an auto-regressive language model and has been pretrained on 2 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over one million new human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.
The data updation has been more recent, up to July 2023 which makes it more accurate with the trends and search.
As per our problem statement, the fine-tuned model is optimized for dialogue use-cases and in human evaluations for helpfulness and safety, it performs really well. The largest model (70B) uses a feature called Grouped-Query Attention (GQA) to enhance inference scalability.
 The use of the transformer architecture, along with specific techniques like SFT and RLHF, indicates a high level of sophistication in the model's development.
I have trained Llama-2 on the question-answering task using the AWS sagemaker. 
Here are more details about the model - 
https://huggingface.co/TheBloke/Llama-2-70B-Chat-fp16


## Fine Tuning : 

I have explored various fine-tuning techniques like multi-task instruction fine-tuning and parameter efficient fine-tuning using different techniques.
Accordiong to our current usecase and the LLM, parameter efficient fine tuning will be the most accurate as it updates only a small subset of task-specific adapter layers and parameters, making it less memory-intensive.Also, PEFT is less prone to the issue of catastrophic forgetting, which can occur during full fine-tuning. There are various methods available for parameter efficient fine-tuning, each with trade-offs in parameter efficiency, memory efficiency, training speed, model quality, and inference costs.
Furthermore, i have applied LoRA to the self-attention layers of the model for significant savings in trainable parameters.
QLoRA is a practice that combines LoRA with quantization techniques to pretrain  model to 4 bits and attach small “Low-Rank Adapters” which are fine-tuned.
 This enables fine-tuning of models with up to 65 billion parameters on a single GPU; despite its efficiency, QLoRA matches the performance of full-precision fine-tuning and achieves state-of-the-art results on language tasks.

LLMs have been associated with problems such as generating toxic language, responding in aggressive tones, and providing potentially harmful information. These issues arise because the models are trained on vast amounts of text data from the Internet, which includes instances of such problematic language.
The tuned versions of Llama-2 use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety which solves our issue. RLHF uses a reward model, with two classes: 'notate' (positive) and 'hate' (negative). PPO is a popular method in RLHF, aiming to align LLM behavior with human preferences.

Further, to monitor upcoming add-ons and customer reviews, I propose the use of Retrieval Augmented Generation (RAG)  that leverage external data sources.  It addresses limitations of LLMs and helps them update their understanding of the world, , addressing the knowledge cutoff issue.
It also prevents the chatbot from providing incorrect information when it doesn't have the answers. Tools like Langchain can assist with this.


RAG processes small chunks of external data through the LLM to create embedding vectors for each. These new representations can be stored in structures called vector stores, enabling fast searching and efficient identification of semantically related text.
