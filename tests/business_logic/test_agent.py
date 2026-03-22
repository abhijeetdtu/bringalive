import pytest

from bringalive.business_logic.agent import Agent
from bringalive.business_logic.logger import logging

logger = logging.getLogger("tests.business_logic.test_agent")


def test_agent_prompt_invoke(default_model):
    logger.info("creating agent")
    agent = Agent(default_model, True, True)
    resp = agent.invoke("Where was gandhi born")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True


def test_agent_prompt_invoke_no_thinking(default_model):
    logger.info("creating agent")
    agent = Agent(default_model, False, True)
    resp = agent.invoke("Where was gandhi born")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True


def test_agent_prompt_invoke_no_thinking_with_system_prompt(default_model, environment_prompt):
    logger.info("creating agent")
    agent = Agent(default_model, False, True, system_prompt=environment_prompt)
    resp = agent.invoke("I move ahead two steps")
    for json in resp:
        print(json, end="\n", flush=True)
    assert True
