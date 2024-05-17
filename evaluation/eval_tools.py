from pymongo import MongoClient
import numpy as np
import seaborn
import matplotlib.pyplot as plt

class evaluation_tools:
    def __init__(self, db_link: str, db_cluster_name: str):
        self.cluster = MongoClient(db_link)
        self.db = self.cluster[db_cluster_name]

    def get_round_data(self, game_id: int, round_id: int):
        """
        Get the data of a round in a game
        :param game_id: The id of the game
        :param round_id: The id of the round
        :return: A dictionary containing the data of the round
        """
        round = self.db["rounds"].find_one({'game_id':game_id,'round_id':round_id})
        if round:
            llm_knows, true_def, deception, correct_guesse = 0, 0, 0, 0
            for player in round['players']:
                if round['player_definitions'][str(player)]['llm_knows_one_of_the_defs']:
                    llm_knows+=1
                if round['player_definitions'][str(player)]['judge_decision']:
                    true_def+=1
                else:
                    if round['votes'][str(player)] == -1:
                        correct_guesse += 1
                    elif round['votes'][str(player)] != player:
                        deception += 1
            round_data = {'llm_knows_ratio': llm_knows/len(round['players']),
                        'true_def_ratio': true_def/len(round['players']),
                        'deception_ratio': deception/(len(round['players']) - true_def) if len(round['players']) - true_def != 0 else -1,
                        'correct_guesse_ratio': correct_guesse/(len(round['players']) - true_def) if len(round['players']) - true_def != 0 else -1,}
        else:
            round_data = None
        return round_data
    
    def get_game_data(self, game_id: int):
        """
        Get the data of a game
        :param game_id: The id of the game
        :return: A dictionary containing the data of the game
        """
        game = self.db["games"].find_one({'game_id':game_id})
        game_data = {'llm_knows_ratio': [],
                      'true_def_ratio': [],
                      'deception_ratio': [],
                      'correct_guesse_ratio': []}
        if game:
            for round in range(1,game['number_of_rounds']+1):
                round_data = self.get_round_data(game_id, round)
                if round_data != None:
                    game_data['llm_knows_ratio'].append(round_data['llm_knows_ratio'])
                    game_data['true_def_ratio'].append(round_data['true_def_ratio'])
                    game_data['deception_ratio'].append(round_data['deception_ratio'])
                    game_data['correct_guesse_ratio'].append(round_data['correct_guesse_ratio'])
                else:
                    print("Round", round, "data is missing")
                    game_data = None
                    break
        else:
            game_data = None
        
        return game_data
    
    def get_word_experiment_data(self, start_game_id: int, end_game_id: int, word_filter: list[tuple[str, str]]):
        """
        Get the data of a word category experiment
        :param start_game_id: The id of the first game
        :param end_game_id: The id of the last game
        :param word_filter: The data filter used in the experiment
        :return: A dictionary containing the data of all games in the experiment
        """
        experiment_data = {}
        for filter in word_filter:
            experiment_data[filter] = {}
        for i in range(start_game_id, end_game_id+1):
            game = self.db["games"].find_one({'game_id':i})
            for filter in word_filter:
                if game['filter_words'] == filter[1] and game['words_file'] == filter[0]:
                    game_data = self.get_game_data(i)
                    if game_data != None:
                        experiment_data[filter][i] = game_data
                    else:
                        print("Game", i, "data is missing")
        
        return experiment_data
    
    def get_rule_experiment_data(self, start_game_id: int, end_game_id: int, true_def_points: list[int]):
        """
        Get the data of a rule set experiment
        :param start_game_id: The id of the first game
        :param end_game_id: The id of the last game
        :param true_def_points: award received for generating true definition in the experiment
        :return: A dictionary containing the data of all games in the experiment
        """
        experiment_data = {}
        for points in true_def_points:
            experiment_data[points] = {}
        for i in range(start_game_id, end_game_id+1):
            game = self.db["games"].find_one({'game_id':i})
            for points in true_def_points:
                if game['correct_definition_points'] == points:
                    game_data = self.get_game_data(i)
                    if game_data != None:
                        experiment_data[points][i] = game_data
                    else:
                        print("Game", i, "data is missing")
        
        return experiment_data
    
    def experiment_game_average(self, experiment_data: dict):
        """
        Get the average data of an experiment on game level
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :return: A dictionary containing the average data of the experiment
        """
        average_data = {'llm_knows_ratio': {key: np.mean(value['llm_knows_ratio']) for key, value in experiment_data.items()},
                        'true_def_ratio': {key: np.mean(value['true_def_ratio']) for key, value in experiment_data.items()},
                        'deception_ratio': {key: np.mean([i for i in value['deception_ratio'] if i!=-1]) for key, value in experiment_data.items()},
                        'correct_guesse_ratio': {key: np.mean([i for i in value['correct_guesse_ratio'] if i!=-1]) for key, value in experiment_data.items()}}
        
        overal_averge_data = {'llm_knows_ratio': {'mean': np.mean(list(average_data['llm_knows_ratio'].values())),
                                                  'std': np.std(list(average_data['llm_knows_ratio'].values()))},
                              'true_def_ratio': {'mean': np.mean(list(average_data['true_def_ratio'].values())),
                                                  'std': np.std(list(average_data['true_def_ratio'].values()))},
                              'deception_ratio': {'mean': np.mean(list(average_data['deception_ratio'].values())),
                                                  'std': np.std(list(average_data['deception_ratio'].values()))},
                              'correct_guesse_ratio': {'mean': np.mean(list(average_data['correct_guesse_ratio'].values())),
                                                  'std': np.std(list(average_data['correct_guesse_ratio'].values()))}}
        return average_data, overal_averge_data
    
    def experiment_round_average(self, experiment_data: dict):
        """
        Get the average data of an experiment on round level
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :return: A dictionary containing the average data of the experiment
        """
        n = len(experiment_data[list(experiment_data.keys())[0]]['llm_knows_ratio'])
        avg_true_def = np.zeros(n)
        std_true_def = np.zeros(n)
        avg_llm_knows = np.zeros(n)
        std_llm_knows = np.zeros(n)
        for i in range(n):
            temp1 = []
            temp2 = []
            for key in experiment_data.keys():
                temp1.append(experiment_data[key]['true_def_ratio'][i])
                temp2.append(experiment_data[key]['llm_knows_ratio'][i])
            avg_true_def[i] = np.mean(temp1)
            std_true_def[i] = np.std(temp1)
            avg_llm_knows[i] = np.mean(temp2)
            std_llm_knows[i] = np.std(temp2)
        
        return {'llm_knows_ratio': {'mean': avg_llm_knows, 'std': std_llm_knows},
                'true_def_ratio': {'mean': avg_true_def, 'std': std_true_def}}
        
    
    
    def plot_true_def_ratio(self, experiment_data: dict, word_filter: str, word_file: str, window_size: int):
        """
        Plot the true definition ratio of an experiment
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :param word_filter: The data filter used in the experiment
        :param window_size: The size of the window used in the moving average
        """
        round_average = self.experiment_round_average(experiment_data)
        avg_true_def_smooth = np.convolve(round_average['true_def_ratio']['mean'], np.ones(window_size), 'valid') / window_size
        avg_llm_knows_smooth = np.convolve(round_average['llm_knows_ratio']['mean'], np.ones(window_size), 'valid') / window_size
        std_true_def_smooth = np.convolve(round_average['true_def_ratio']['std'], np.ones(window_size), 'valid') / window_size
        std_llm_knows_smooth = np.convolve(round_average['llm_knows_ratio']['std'], np.ones(window_size), 'valid') / window_size
        n = len(round_average['true_def_ratio']['mean'])
        # set xlabal as round number and y label as ratio

        plt.figure()
        plt.plot(avg_true_def_smooth, label='True Definition Ratio')
        plt.fill_between(range(n+1-window_size), avg_true_def_smooth-std_true_def_smooth, 
                        avg_true_def_smooth+std_true_def_smooth, alpha=0.2)
        plt.plot(avg_llm_knows_smooth, label='LLM Knows Ratio')
        plt.fill_between(range(n+1-window_size), avg_llm_knows_smooth-std_llm_knows_smooth, 
                        avg_llm_knows_smooth+std_llm_knows_smooth, alpha=0.2)

        plt.xlabel('Rounds')
        plt.ylabel('True Definiton Ratio')
        plt.title(word_filter + ' ' + word_file)
        plt.legend()
        plt.show()
        
        
        
