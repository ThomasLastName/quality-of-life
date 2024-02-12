
from email import message
from quality_of_life.my_base_utils import load_txt, support_for_progress_bars
from quality_of_life.ansi import bcolors

import sys
from openai import OpenAI


#
# ~~~ https://stackoverflow.com/a/10446010/11595884
class Meta(type):
    def __contains__(cls, item):
        return item in cls.set

class BaseSequence(metaclass=Meta):
    pass

class EndSequences(BaseSequence):
    set = set([ "Exit", "exit", "Quit", "quit", "q" ])
    @classmethod
    def append(cls,new_item):
        cls.set = cls.set | set(new_item)



def format_messages(
        list_of_messages,
        user_name = "User",
        assistant_name = "Bot",
        role_color = bcolors.HEADER,
        content_color = (sys.stdout.main_color if hasattr(sys.stdout,"main_color") else bcolors.OKGREEN),
        end = "print",
        buffer = 1,
        suffix = ":",
        separator = " ",
        between_lines = "\n"+bcolors.OKBLUE+"--------------\n"
    ):
    #
    # ~~~ Boiler plate stuff
    assert end=="print" or end=="return"
    user_name = "user" if (user_name is None) else user_name
    assistant_name = "Bot" if (assistant_name is None) else assistant_name
    expected_roles = [ user_name, assistant_name ]
    just_val = max([ len(role) for role in expected_roles ]) + len(suffix)
    formatted_output = ""
    #
    # ~~~ Loop through the list `list_of_messages` and execute the relevant logic on each item
    with support_for_progress_bars():
        for j,dictionary in enumerate(list_of_messages):
            if j>0:
                formatted_output += between_lines
            formatted_output += format_line(
                role = dictionary["role"],
                content = dictionary["content"],
                just_val = just_val,
                user_name = user_name,
                assistant_name = assistant_name,
                role_color = role_color,
                content_color = content_color,
                buffer = buffer,
                suffix = suffix,
                separator = separator
            )
        #
        # ~~~ Output
        formatted_output = formatted_output.strip("\n")   # ~~~ we don't actually want that last line breack
        if end=="return":
            return formatted_output
        else:
            print(formatted_output)


def format_line(
        role,
        content,
        just_val,
        user_name = "User",
        assistant_name = "Bot",
        role_color = bcolors.HEADER,
        content_color = (sys.stdout.main_color if hasattr(sys.stdout,"main_color") else bcolors.OKGREEN),
        buffer = 0,
        suffix = ":",
        separator = " "
    ):
    if role=="user" and user_name is not None:
        role = user_name
    if role=="assistant" and assistant_name is not None:
        role = assistant_name
    name = role_color+(role+suffix).rjust(just_val)
    text = content_color+content
    space_in_between = " "*buffer+bcolors.HEADER+separator+" "*buffer
    return name + space_in_between + text


def compile_system_prompt(list_of_text_strings):
    return [
            { "role":"system", "content":text_string }
            for text_string in list_of_text_strings
        ]


def interactive_chatbot(
        completer,
        system_prompt,
        end_sequences = EndSequences,
        user_prompt = "You",
        assistant_name = "Bot",
        role_color = bcolors.HEADER,
        content_color = (sys.stdout.main_color if hasattr(sys.stdout,"main_color") else bcolors.OKGREEN),
        buffer = 0,
        suffix = ":",
        separator = " ",
        between_lines = bcolors.OKBLUE+"-------------------"
    ):
    #
    # ~~~ Initialization
    system_prompt = system_prompt if isinstance( system_prompt[0], dict ) else compile_system_prompt(system_prompt)
    message_log = system_prompt
    expected_tags = [ user_prompt, assistant_name ]
    just_val = max([ len(role) for role in expected_tags ]) + len(suffix)
    #
    # ~~~ Wrap a frequently called text formatting util
    def my_format( role, content ):
        return format_line(
                role,
                content,
                just_val = just_val,
                assistant_name = assistant_name,
                role_color = role_color,
                content_color = content_color,
                buffer = buffer,
                suffix = suffix,
                separator = separator
            )
    #
    # ~~~ Implement the actual logic of the conversation between user and bot
    while True:
        #
        # ~~~ Get the input from the user, print it in the terminal and update `messages` to include the bot response
        user_input = input(my_format( role=user_prompt, content="" ))  # ~~~ both gets the input and prints it, too
        message_log.append({ "role":"user", "content":user_input })    # ~~~ add it to the record
        print(between_lines)
        #
        # ~~~ If user says `exit` or one of the other end_sequences, exit the loop
        if user_input in end_sequences:
            print(my_format( role="Interrupt Sequence", content=user_input ))
            break
        #
        # ~~~ Get the response from the bot, print it in the terminal, and update `message_log` to include the last bot response
        bot_response = completer(message_log)                               # ~~~ gets the response
        print(my_format( role="assistant", content=bot_response ))          # ~~~ print it
        message_log.append({ "role":"assistant", "content":bot_response })  # ~~~ add it to the record
        print(between_lines)
    #
    # ~~~ return the log of events; try, e.g., `format_messages(message_log)`
    return message_log

#