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
        player_data_for_db = player_data.copy()
        player_data_for_db.pop("logger", None)
        player_data_for_db.pop("llm", None)
        self.db["players"].insert_one(player_data_for_db)

    def get_last_player_id(self) -> int:
        last_player = self.db["players"].find_one(sort=[("_id", -1)])
        if last_player:
            return last_player["player_id"]
        return 0

    def update_player(self, player_id: int, update_data: dict) -> None:
        self.db["players"].update_one({"player_id": player_id}, {"$set": update_data})

    def get_last_game_id(self) -> int:
        last_game = self.db["games"].find_one(sort=[("game_id", -1)])
        if last_game:
            return last_game["game_id"]
        return 0

    def insert_game(self, game_data: dict) -> None:
        # Check if the game_data has "logger" key, drop it
        game_data_for_db = game_data.copy()
        game_data_for_db.pop("logger", None)
        self.db["games"].insert_one(game_data_for_db)

    def get_player_rounds(self, player_id: int, window_size: int) -> list:
        """
        Rounds collection in the mongodb has a column called players which is an array of player_ids sorted by the round_id columns.
        This function sorts the rounds by round_id and returns the rounds where the player_id is in the players array for the last window_size games.
        if window_size is -1, it returns all the rounds where the player_id is in the players array.
        """
        if window_size != -1:
            return list(
                self.db["rounds"].find({"players": player_id}).sort([("round_id", -1)]).limit(window_size)
            )[::-1]
        return list(self.db["rounds"].find({"players": player_id}).sort([("round_id", 1)]))

    def insert_round(self, round_data: dict) -> None:
        # Check if the round_data has "logger" key, drop it
        round_data_for_db = round_data.copy()
        round_data_for_db.pop("logger", None)
        round_data_for_db.pop("definitions_permutation", None)
        # player_definitions and votes are dictionaries, convert all keys to strings
        player_definitions = {str(k): v for k, v in round_data_for_db["player_definitions"].items()}
        votes = {str(k): v for k, v in round_data_for_db["votes"].items()}
        scores = {str(k): v for k, v in round_data_for_db["scores"].items()}
        round_data_for_db["player_definitions"] = player_definitions
        round_data_for_db["votes"] = votes
        round_data_for_db["scores"] = scores
        self.db["rounds"].insert_one(round_data_for_db)

    def send_ping(self) -> None:
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command("ping")
            self.logger.info("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            self.logger.error(f"Error connecting to MongoDB: {e}")
            raise e
