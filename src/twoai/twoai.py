import json
import ollama
from colorama import Fore, Style
from . import AgentDetails, Message
import logging

# Initialize logger with file handler
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='twoai.log',
                    filemode='a')
logger = logging.getLogger(__name__)

class TWOAI:
    """
    Class representing an AI that can engage in a conversation with another AI.
    
        ai_details (AIDetails): Details of the AI including name and objective.
        model (str): The model used by the AI.
        system_prompt (str): The prompt for the AI conversation system.
        task (str): The task assigned by the user.
        max_tokens (int): The maximum number of tokens to generate in the AI response.
        num_context (int): The number of previous messages to consider in the AI response.
        extra_stops (list): Additional stop words to include in the AI response.
        exit_word (str): The exit word to use in the AI response. Defaults to "<DONE!>".
        max_exit_words (int): The maximum number of exit words to include in the AI responses for the conversation to conclude. Defaults to 2.
    """
    def __init__(
            self, 
            model: str, 
            agent_details: AgentDetails, 
            system_prompt: str, 
            task: str,
            max_tokens: int=4094, 
            num_context: int=4094, 
            extra_stops: list[str] = [],
            exit_word: str = "<DONE!>",
            temperature: int = 0.7,
            max_exit_words: int = 2
        ) -> None:
        logger.info("Initializing TWOAI instance")
        self.agent_details = agent_details
        self.model = model
        self.system_prompt = system_prompt
        self.task = task
        self.max_tokens = max_tokens
        self.num_context = num_context
        self.extra_stops = extra_stops
        self.temperature = temperature
        
        self.messages = ""
        self.current_agent = agent_details[0]

        self.exit_word = exit_word
        self.exit_word_count = 0
        self.max_exit_words = max_exit_words

        self.conversation = []
        system_message = Message(from_="system", value=agent_details[0]['instructions'])
        self.conversation.append(system_message.dict())
        user_message = Message(from_="user", value=str(self.task))
        self.conversation.append(user_message.dict())

    def bot_say(self, msg: str, color: str = Fore.LIGHTGREEN_EX):
        logger.debug(f"Bot says: {msg}")  # Changed from info to debug
        print(color + msg.strip() + "\t\t" + Style.RESET_ALL )

    def get_reactor_ai(self) -> AgentDetails:
        if self.current_agent['name'] == self.agent_details[0]['name']:
            return self.agent_details[1]
        return self.agent_details[0]

    def __get_updated_template_str(self):
        result = self.system_prompt.format(
            actor_name=self.current_agent['name'],
            reactor_name=self.get_reactor_ai()["name"],
            instructions=self.current_agent.get('instructions', ''),
            task = self.task,       
            schema=self.current_agent.get('schema', '')
        )
        return result

    def __show_cursor(self):
        print("\033[?25h", end="")

    def __hide_cursor(self):
        print('\033[?25l', end="")

    def next_response(self, show_output: bool = False) -> str:
        if len(self.agent_details) < 2:
            logger.error("Not enough AI details provided")
            raise Exception("Not enough AI details provided")

        reactor_ai = self.get_reactor_ai()
        instructions = self.__get_updated_template_str()
        convo = f"""
        {instructions}

        {self.messages}
        """

        current_model = self.model
        if model := self.current_agent.get('model', None):
            current_model = model

        if show_output:
            self.__hide_cursor()
            logger.debug(f"{self.current_agent['name']} is thinking...")  # Changed from info to debug
            print(Fore.YELLOW + f"{self.current_agent['name']} is thinking..." + Style.RESET_ALL, end='\r')

        resp = ollama.generate(
            model=current_model, 
            prompt=convo.strip(), 
            stream=False, 
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "num_ctx": self.num_context,
                "stop": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "###",
                    "\r\n",
                   f"{reactor_ai['name']}: " if self.current_agent['name'] != reactor_ai['name'] else f"{self.current_agent['name']}: "
                    
                ] + self.extra_stops
            }
        )

        text: str = resp['response'].strip()
        if not text:
            logger.error(f"{self.current_agent['name']} response was empty, trying again.")
            return self.next_response(show_output)

        if not text.startswith(self.current_agent['name'] + ": "):
            text = self.current_agent['name'] + ": " + text

        message = Message(from_=self.current_agent['name'], value=text)
        self.conversation.append(message.dict())
        self.messages += text + "\n"

        if show_output:
            if self.agent_details.index(self.current_agent) == 0:
                self.bot_say(text)
            else:
                self.bot_say(text, Fore.BLUE)
        
        self.current_agent = self.get_reactor_ai()
        self.__show_cursor()
        return text

    def start_conversation(self):
        try:
            while True:
                res = self.next_response(show_output=True)
                if self.exit_word in res:
                    self.exit_word_count += 1
                if self.exit_word_count == self.max_exit_words:
                    print(Fore.RED + "The conversation was concluded..." + Style.RESET_ALL)
                    with open('conversation_logs.jsonl', 'a') as file:
                        file.write(json.dumps(self.conversation) + '\n')
                    self.__show_cursor()
                    return
        except KeyboardInterrupt:
            print(Fore.RED + "Closing Conversation..." + Style.RESET_ALL)
            self.__show_cursor()
            return
