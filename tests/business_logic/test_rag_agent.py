from bringalive.business_logic.agent import RAGAgent
import pytest

from bringalive.business_logic.agent import BehaviorAgent
from bringalive.business_logic.logger import logging

logger = logging.getLogger("tests.business_logic.test_rag_agent")


def test_rag_agent_prompt_invoke(default_collection, default_model):
    logger.info("creating rag agent")
    agent = RAGAgent(default_collection, default_model, True, True)
    logger.info("invoking rag agent")
    resp = agent.invoke("Where was gandhi born")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True


def test_rag_agent_prompt_invoke_with_behavior(default_collection, default_model, behavior_prompt):
    logger.info("creating rag agent")
    agent = RAGAgent(default_collection, default_model, True, True,
                     system_prompt=behavior_prompt)
    logger.info("invoking rag agent")
    resp = agent.invoke("Where were you born")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True


def test_rag_agent_prompt_invoke_with_behavior_no_thinking(default_collection, default_model, behavior_prompt):
    logger.info("creating rag agent")
    agent = RAGAgent(default_collection, default_model, False, True,
                     system_prompt=behavior_prompt)
    logger.info("invoking rag agent")
    resp = agent.invoke("Where were you born")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True


def test_behavior_agent_prompt_invoke_no_thinking_q1(default_collection, behavior_prompt):
    logger.info("creating rag agent")
    agent = BehaviorAgent(default_collection, "deepseek-r1:8b", False, True,
                          system_prompt=behavior_prompt)
    logger.info("invoking rag agent")
    resp = agent.invoke("Where were you born")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True
