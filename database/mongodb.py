from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from utils.logger import setup_logger


class MongoDB:
    def __init__(self, connection_string: str) -> None:
        self.logger = setup_logger("mongodb", "logs/mongodb.log")
        self.client = MongoClient(connection_string, server_api=ServerApi("1"))
        self.send_ping()
        self.db = self.client["balderdash_game"]

    def insert_player(self, player_data: dict) -> None:
        # Check if the player_data has "logger," "device," "tokenizer," "model" keys, drop them
        player_data.pop("logger", None)
        player_data.pop("device", None)
        player_data.pop("tokenizer", None)
        player_data.pop("model", None)
        self.db["players"].insert_one(player_data)

    def update_player(self, player_id: int, update_data: dict) -> None:
        self.db["players"].update_one({"_id": player_id}, {"$set": update_data})

    def insert_round(self, round_data: dict) -> None:
        # Check if the round_data has "logger" key, drop it
        round_data.pop("logger", None)
        self.db["rounds"].insert_one(round_data)

    def send_ping(self) -> None:
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command("ping")
            self.logger.info("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            self.logger.error(f"Error connecting to MongoDB: {e}")
            raise e
