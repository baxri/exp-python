import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from exp-python!")
    print(f"OPEN_ROUTER_API_KEY = {os.environ.get('OPEN_ROUTER_API_KEY')}")

if __name__ == "__main__":
    main()
