from utils.logger import setup_logger


class Round:
    def __init__(self, round_id: int, word: str, correct_definition: str, pos: str) -> None:
        self.logger = setup_logger(f"round_{round_id}", f"logs/round_{round_id}.log")
        self.logger.info(f"Initializing Round: {round_id} - Word: {word}")
        self.round_id = round_id
        self.word = word
        self.correct_definition = correct_definition
        self.pos = pos
        self.player_definitions = {}
        self.votes = {}

    def add_player_definition(self, player_id: int, definition: str) -> None:
        self.logger.info(f"Adding player definition: {player_id} - {definition}")
        self.player_definitions[player_id] = definition

    def add_vote(self, player_id: int, vote: int) -> None:
        self.logger.info(f"Adding vote: {player_id} - {vote}")
        self.votes[player_id] = vote

    def calculate_scores(self):
        self.logger.info("Calculating scores")
        scores = {}
        correct_votes = 0

        # Count the number of votes for each definition
        vote_counts = {}
        for vote in self.votes.values():
            if vote in vote_counts:
                vote_counts[vote] += 1
            else:
                vote_counts[vote] = 1

        # Assign scores to players
        for player_id, definition in self.player_definitions.items():
            if definition == self.correct_definition:
                scores[player_id] = 2  # Player gets 2 points for the correct definition
                correct_votes += 1
            else:
                scores[player_id] = vote_counts.get(
                    player_id, 0
                )  # Player gets points based on the votes received

        # If no one guessed the correct definition, the dasher gets 2 points
        if correct_votes == 0:
            scores["dasher"] = 2

        self.logger.info(f"Calculated scores: {scores}")
        return scores
