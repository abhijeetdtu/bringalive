import requests
import json

from bringalive.business_logic.logger import logging
from bringalive.business_logic.memory import Memory

logger = logging.getLogger("bringalive.business_logic.agent")


class Agent:

    def __init__(self, model, think, stream, system_prompt=None,
                 end_point="http://localhost:11434/api/chat"):
        self.model = model
        self.think = think
        self.stream = stream
        self.system_prompt = system_prompt
        self.end_point = end_point

    def get_system_prompt(self):
        return self.system_prompt

    def set_system_prompt(self, new_system_prompt):
        self.system_prompt = new_system_prompt

    def add_to_system_prompt(self, new_system_prompt):
        if self.system_prompt is not None:
            self.system_prompt += new_system_prompt
        else:
            self.set_system_prompt(new_system_prompt)

    def get_response(self, prompt, think=None, stream=None, options={}):
        logger.info("got prompt %s", prompt)
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system", "content": self.system_prompt
            })
        messages.append({
            "role": "user",
            "content": prompt
        })
        think = think if think is not None else self.think
        stream = stream if stream is not None else self.stream
        payload = {"model": self.model, "messages": messages,
                   "think": think, "stream": stream, "options": options}

        logger.info("submitting payload %s", payload)
        return requests.post(self.end_point, json=payload, stream=self.stream)

    def get_stream(self, response):
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                yield json.loads(decoded_line)["message"]

    def invoke(self, prompt, think=None, stream=None, options={}):
        resp = self.get_response(prompt, think, stream, options)
        return self.get_stream(resp)


class RAGAgent(Agent):

    def __init__(self, collection_name, model, think, stream,
                 system_prompt=None,
                 end_point="http://localhost:11434/api/chat"):
        super().__init__(model, think, stream, system_prompt, end_point)
        self.collection_name = collection_name

    def get_response(self, prompt, think=None, stream=None, options={}):
        query_results = "\n".join(
            Memory(collection=self.collection_name).query_documents(prompt))
        logger.info("got query results %s", query_results)
        new_prompt = f"""Question : {prompt}
        Context : Use the following facts only, to respond accurately : {query_results} 
        Answer:"""
        # self.add_to_system_prompt(additional_context)
        return super().get_response(new_prompt, think, stream, options)


class BehaviorAgent(Agent):

    def __init__(self, collection_name, model, think, stream,
                 system_prompt=None,
                 end_point="http://localhost:11434/api/chat"):
        super().__init__(model, think, stream, system_prompt, end_point)
        self.collection_name = collection_name

    def get_response(self, prompt, think=None, stream=None, options={}):
        resp = super().get_response(
            f"How would you go about responding to the following : {prompt}",
            think=False, stream=False, options={"num_predict": 128})
        query = self.get_stream(resp)
        query = "\n".join([q["content"] for q in query])
        logger.info("query %s", query)
        query_results = "\n".join(
            Memory(collection=self.collection_name).query_documents(query))
        logger.info("got query results %s", query_results)
        additional_context = f"Use the following facts only if helpful, to respond accurately : {query_results} : Give your personal opinion"
        self.add_to_system_prompt(additional_context)
        return super().get_response(prompt, think, stream, options)
