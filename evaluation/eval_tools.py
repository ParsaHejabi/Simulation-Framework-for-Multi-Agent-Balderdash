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
            player_types = {}
            for player in round['players']:
                player_data = self.db["players"].find_one({'player_id':player})
                try:
                    player_types[player] = player_data['llm_name']
                except:
                    player_types[player] = player_data['name'][11:]
            
            llm_knows, true_def, deception, correct_guesse, scores = {}, {}, {}, {}, {}
            for player in round['players']:
                llm_knows[player_types[player]] = 0
                true_def[player_types[player]] = 0
                deception[player] = 0
                correct_guesse[player_types[player]] = 0
                scores[player_types[player]] = 0
            
            for player in round['players']:
                scores[player_types[player]] += round['scores'][str(player)]
                if round['player_definitions'][str(player)]['llm_knows_one_of_the_defs']:
                    llm_knows[player_types[player]] += 1
                if round['player_definitions'][str(player)]['judge_decision']:
                    true_def[player_types[player]] += 1
                else:
                    if round['votes'][str(player)] == -1:
                        correct_guesse[player_types[player]] += 1
                    elif round['votes'][str(player)] != player:
                        deception[round['votes'][str(player)]] += 1
            
            voting_players_number = len(round['players']) - sum([value for value in true_def.values()]) - 1
            player_types_inverse = {}
            for k, v in player_types.items():
                player_types_inverse.setdefault(v, []).append(k)
                deception[k] = deception[k]/voting_players_number if voting_players_number > 0 else -1
            
            
            round_data = {'llm_knows_ratio': {},
                        'true_def_ratio': {},
                        'deception_ratio': {},
                        'correct_guesse_ratio': {},
                        'scores_ratio': {}}
            
            for llm in list(player_types_inverse.keys()):
                round_data['llm_knows_ratio'][llm] = llm_knows[llm]/len(player_types_inverse[llm])
                round_data['true_def_ratio'][llm] = true_def[llm]/len(player_types_inverse[llm])
                deceptions = [deception[player] for player in player_types_inverse[llm] if deception[player] != -1]
                round_data['deception_ratio'][llm] = np.mean(deceptions) if len(deceptions) > 0 else -1
                round_data['correct_guesse_ratio'][llm] = correct_guesse[llm]/(len(player_types_inverse[llm]) - true_def[llm]) if len(player_types_inverse[llm]) - true_def[llm] != 0 else -1
                round_data['scores_ratio'][llm] = scores[llm]/len(player_types_inverse[llm])
            
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
                      'correct_guesse_ratio': [],
                      'scores_ratio': []}
        if game:
            for round in range(1,game['number_of_rounds']+1):
                round_data = self.get_round_data(game_id, round)
                if round_data != None:
                    game_data['llm_knows_ratio'].append(round_data['llm_knows_ratio'])
                    game_data['true_def_ratio'].append(round_data['true_def_ratio'])
                    game_data['deception_ratio'].append(round_data['deception_ratio'])
                    game_data['correct_guesse_ratio'].append(round_data['correct_guesse_ratio'])
                    game_data['scores_ratio'].append(round_data['scores_ratio'])
                else:
                    print("Round", round, "data is missing")
                    game_data = None
                    break
        else:
            game_data = None
        
        return game_data
    
    def get_experiment_data(self, start_game_id: int, end_game_id: int):
        """
        Get the data of a word category experiment
        :param start_game_id: The id of the first game
        :return: A dictionary containing the data of all games in the experiment
        """
        experiment_data = {}
        for i in range(start_game_id, end_game_id+1):
            game = self.db["games"].find_one({'game_id':i})
            game_data = self.get_game_data(i)
            if game_data != None:
                experiment_data.setdefault((game['words_file'], 
                                            game['filter_words'],
                                            game['receiving_vote_points'], 
                                            game['correct_vote_points'],
                                            game['correct_definition_points']), {})[i] = game_data
            else:
                print("Game", i, "data is missing")
        
        return experiment_data
    
    
    def experiment_game_average(self, experiment_data: dict):
        """
        Get the average data of an experiment on game level
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :return: A dictionary containing the average data of the experiment
        """
        llm_players = list(experiment_data.values())[0]['llm_knows_ratio'][0].keys()
        average_data = {}
        overal_averge_data = {}
        for llm in llm_players:
            average_data[llm] = {'llm_knows_ratio': {key: np.mean([ratio[llm] for ratio in value['llm_knows_ratio']]) for key, value in experiment_data.items()},
                            'true_def_ratio': {key: np.mean([ratio[llm] for ratio in value['true_def_ratio']]) for key,
                                                value in experiment_data.items()},
                            'deception_ratio': {key: np.mean([i for i in [ratio[llm] for ratio in value['deception_ratio']] if i!=-1]) for key,
                                                value in experiment_data.items()},
                            'correct_guesse_ratio': {key: np.mean([i for i in [ratio[llm] for ratio in value['correct_guesse_ratio']] if i!=-1]) for key,
                                                        value in experiment_data.items()},
                            'scores_ratio': {key: np.mean([ratio[llm] for ratio in value['scores_ratio']]) for key, value in experiment_data.items()}}
        
            overal_averge_data[llm] = {'llm_knows_ratio': {'mean': np.mean(list(average_data[llm]['llm_knows_ratio'].values())),
                                                        'std': np.std(list(average_data[llm]['llm_knows_ratio'].values()))},
                                    'true_def_ratio': {'mean': np.mean(list(average_data[llm]['true_def_ratio'].values())),
                                                        'std': np.std(list(average_data[llm]['true_def_ratio'].values()))},
                                    'deception_ratio': {'mean': np.mean(list(average_data[llm]['deception_ratio'].values())),
                                                        'std': np.std(list(average_data[llm]['deception_ratio'].values()))},
                                    'correct_guesse_ratio': {'mean': np.mean(list(average_data[llm]['correct_guesse_ratio'].values())),
                                                        'std': np.std(list(average_data[llm]['correct_guesse_ratio'].values()))},
                                    'scores_ratio': {'mean': np.mean(list(average_data[llm]['scores_ratio'].values())),
                                                        'std': np.std(list(average_data[llm]['scores_ratio'].values()))}}
        return average_data, overal_averge_data
    
    def experiment_round_average(self, experiment_data: dict):
        """
        Get the average data of an experiment on round level
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :return: A dictionary containing the average data of the experiment
        """
        llm_players = list(list(experiment_data.values())[0]['llm_knows_ratio'][0].keys())
        n = len(experiment_data[list(experiment_data.keys())[0]]['llm_knows_ratio'])
        avg_true_def = {llm: np.zeros(n) for llm in llm_players}
        std_true_def = {llm: np.zeros(n) for llm in llm_players}
        avg_llm_knows = {llm: np.zeros(n) for llm in llm_players}
        std_llm_knows = {llm: np.zeros(n) for llm in llm_players}
        avg_scores = {llm: np.zeros(n) for llm in llm_players}
        std_scores = {llm: np.zeros(n) for llm in llm_players}
        for i in range(n):
            for llm in llm_players:
                temp1 = []
                temp2 = []
                temp3 = []
                for key in experiment_data.keys():
                    temp1.append(experiment_data[key]['true_def_ratio'][i][llm])
                    temp2.append(experiment_data[key]['llm_knows_ratio'][i][llm])
                    temp3.append(experiment_data[key]['scores_ratio'][i][llm])
                avg_true_def[llm][i] = np.mean(temp1)
                std_true_def[llm][i] = np.std(temp1)
                avg_llm_knows[llm][i] = np.mean(temp2)
                std_llm_knows[llm][i] = np.std(temp2)
                avg_scores[llm][i] = np.mean(temp3)
                std_scores[llm][i] = np.std(temp3)
        
        return {'llm_knows_ratio': {'mean': avg_llm_knows, 'std': std_llm_knows},
                'true_def_ratio': {'mean': avg_true_def, 'std': std_true_def},
                'scores_ratio': {'mean': avg_scores, 'std': std_scores}}
        
    
    
    def plot_true_def_ratio(self, experiment_data: dict, word_filter: str, word_file: str, window_size: int):
        """
        Plot the true definition ratio of an experiment
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :param word_filter: The data filter used in the experiment
        :param window_size: The size of the window used in the moving average
        """
        round_average = self.experiment_round_average(experiment_data)
        llm_players = list(list(experiment_data.values())[0]['llm_knows_ratio'][0].keys())
        _, axs = plt.subplots(len(llm_players), 1, figsize=(7, len(llm_players) * 5))
        for i, llm in enumerate(llm_players):
            avg_true_def_smooth = np.convolve(round_average['true_def_ratio']['mean'][llm], np.ones(window_size), 'valid') / window_size
            avg_llm_knows_smooth = np.convolve(round_average['llm_knows_ratio']['mean'][llm], np.ones(window_size), 'valid') / window_size
            std_true_def_smooth = np.convolve(round_average['true_def_ratio']['std'][llm], np.ones(window_size), 'valid') / window_size
            std_llm_knows_smooth = np.convolve(round_average['llm_knows_ratio']['std'][llm], np.ones(window_size), 'valid') / window_size
            n = len(round_average['true_def_ratio']['mean'][llm])
            # set xlabal as round number and y label as ratio

            ax = axs[i] if len(llm_players) > 1 else axs
            ax.plot(avg_true_def_smooth, label='True Definition Ratio')
            ax.fill_between(range(n+1-window_size), avg_true_def_smooth-std_true_def_smooth, 
                            avg_true_def_smooth+std_true_def_smooth, alpha=0.2)
            ax.plot(avg_llm_knows_smooth, label='LLM Knows Ratio')
            ax.fill_between(range(n+1-window_size), avg_llm_knows_smooth-std_llm_knows_smooth, 
                            avg_llm_knows_smooth+std_llm_knows_smooth, alpha=0.2)

            ax.set_xlabel('Rounds')
            ax.set_ylabel('True Definiton Ratio')
            ax.set_title(llm + ' ' + word_filter + ' ' + word_file)
            ax.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_scores_ratio(self, experiment_data: dict, word_filter: str, word_file: str, window_size: int):
        """
        Plot the scores ratio of an experiment
        :param experiment_data: A dictionary containing the data of all games in the experiment
        :param word_filter: The data filter used in the experiment
        :param window_size: The size of the window used in the moving average
        """
        round_average = self.experiment_round_average(experiment_data)
        llm_players = list(list(experiment_data.values())[0]['llm_knows_ratio'][0].keys())
        # plot all scores in one figure
        plt.figure(figsize=(7, 5))
        for llm in llm_players:
            avg_scores_smooth = np.convolve(round_average['scores_ratio']['mean'][llm], np.ones(window_size), 'valid') / window_size
            std_scores_smooth = np.convolve(round_average['scores_ratio']['std'][llm], np.ones(window_size), 'valid') / window_size
            n = len(round_average['scores_ratio']['mean'][llm])
            plt.plot(avg_scores_smooth, label=llm)
            plt.fill_between(range(n+1-window_size), avg_scores_smooth-std_scores_smooth, 
                            avg_scores_smooth+std_scores_smooth, alpha=0.2)
        plt.xlabel('Rounds')
        plt.ylabel('Scores Ratio')
        plt.title('Scores Ratio ' + word_filter + ' ' + word_file)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        
        
