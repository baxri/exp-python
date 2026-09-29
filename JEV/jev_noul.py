import os

from typesafe_sdk import TypeSafeClient, Noul


print(f"OPEN_ROUTER_API_KEY = {os.environ.get('OPEN_ROUTER_API_KEY')}")

client = TypeSafeClient(
    api_key=os.environ.get("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api",
)


res = client.system_one(
    state="I lost my wallet.",
    questions={
        "lost_something": Noul(instructions="Did the user lose something?"),
        "is_angry": Noul(instructions="Is the user angry?"),
    },
)


print("res", res)


# def respond(lost_something):
#     if lost_something:
#         return (
#             "I'm sorry to hear that you lost something. Can you tell me more about it?"
#         )
#     else:
#         return "That's great! I'm glad you didn't lose anything."


# def ask():
#     while True:
#         answer = input("Did you lose something?")

#         res = client.system_one(
#             state=f"{answer}",
#             questions={
#                 "lost_something": Noul(instructions="Did the user lose something?")
#             },
#         )

#         print("res", res)

#         if answer == "Y":
#             return True
#         if answer == "N":
#             return False
#         print('Please answer with "Y" or "N".')


# def main():
#     lost_something = ask()
#     print(respond(lost_something))


# if __name__ == "__main__":
#     main()
