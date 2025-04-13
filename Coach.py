import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.episode_counter = 0  # Track complete games across iterations
        self.episodes_since_save = 0  # Track games since last save
        self.current_iteration_games = []  # List to store complete games for current iteration

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1

            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

            print(f"Turn #{episodeStep}")
            if self.args.verbose:
                canonicalBoard.display()
            if episodeStep % 10 == 0:
                log.info(f"Turn #{episodeStep}")

            temp = int(episodeStep < self.args.tempThreshold)
            

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            #print(f"Action Probabilities: {pi}")
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b.encode(), self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            #print(f"action selected in coach: {action}")
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action, verbose=self.args.verbose)

            r = self.game.getGameEnded(board, self.curPlayer, verbose=self.args.verbose)

            if r != 0:
                log.info(f"The outcome - r value: {r}")
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def savePartialExamples(self, iteration):
        """Save partial examples for the current iteration"""
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Save the number of completed games along with the examples
        save_data = {
            'games': self.current_iteration_games,
            'games_completed': self.episode_counter,
            'checkpoint_id': self.episode_counter
        }
        
        filename = os.path.join(folder, f"partial_iter_{iteration}_{self.episode_counter}.examples")
        log.info(f"Saving partial examples to {filename} with {len(self.current_iteration_games)} games completed")
        with open(filename, "wb+") as f:
            Pickler(f).dump(save_data)
        f.closed

    def loadPartialExamples(self):
        """Load any partial examples from the previous run"""
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            log.info("No checkpoint folder found, starting fresh")
            return
            
        # Find any partial example files
        try:
            partial_files = [f for f in os.listdir(folder) if f.startswith(f"partial_iter_{self.args.starting_iteration}_")]
        except:
            log.info("No partial files found, starting fresh")
            return
            
        if partial_files:
            log.info(f"Found partial example files: {partial_files}")
            
            # Sort files by episode number to get the latest one
            partial_files.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))
            latest_file = partial_files[-1]
            latest_episode = int(latest_file.split("_")[3].split(".")[0])
            
            filepath = os.path.join(folder, latest_file)
            log.info(f"Loading most recent partial examples from {filepath}")
            
            try:
                with open(filepath, "rb") as f:
                    save_data = Unpickler(f).load()
                    
                    # Extract data based on format (for backward compatibility)
                    if isinstance(save_data, dict):
                        self.current_iteration_games = save_data['games']
                        self.episode_counter = save_data['games_completed']
                    else:
                        # Old format - just a list of examples
                        self.current_iteration_games = save_data
                        self.episode_counter = latest_episode
                
                log.info(f"Loaded {len(self.current_iteration_games)} games from partial file")
                log.info(f"Resuming from game {self.episode_counter}")
            except Exception as e:
                log.warning(f"Error loading partial file {latest_file}: {e}")
                self.current_iteration_games = []

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        # Check for and load partial iterations on first run
        if not self.skipFirstSelfPlay:
            self.loadPartialExamples()

        for i in range(self.args.starting_iteration, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                # Calculate how many more games we need to run
                games_to_run = self.args.numEps - len(self.current_iteration_games)
                log.info(f"Need {games_to_run} more games for this iteration")
                
                for _ in tqdm(range(games_to_run), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    episode_start_time = time.time()
                    
                    # Execute a complete game and add to our collection
                    game_examples = self.executeEpisode()
                    self.current_iteration_games.append(game_examples)
                    
                    self.episode_counter += 1
                    self.episodes_since_save += 1
                    episode_end_time = time.time()
                    log.info(f"Game {self.episode_counter} done in {round((episode_end_time - episode_start_time) * 1000)}ms")
                    
                    # Save partial examples every X episodes
                    if self.episodes_since_save >= self.args.saveFrequency:
                        self.savePartialExamples(i)
                        self.episodes_since_save = 0

                # Now that we have all games for this iteration, add them to the training history
                # First, flatten the examples from all games into one list
                iteration_examples = []
                for game in self.current_iteration_games:
                    iteration_examples.extend(game)
                
                self.trainExamplesHistory.append(iteration_examples)
                
                # Clean up any partial files for this iteration as we've completed it
                self.cleanupPartialFiles(i)
                
                # Reset for next iteration
                self.current_iteration_games = []
                self.episodes_since_save = 0

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # In AlphaGo Zero, the new player is accepted if it has a winrate of 55% against the previous version,
            # but in AlphaZero, there is just a single network continuously updated
            if self.args.arenaCompare:
                # training new network, keeping a copy of the old one
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                pmcts = MCTS(self.game, self.pnet, self.args)

                self.nnet.train(trainExamples)
                nmcts = MCTS(self.game, self.nnet, self.args)

                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                            lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                    log.info('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                self.nnet.train(trainExamples)
                log.info(f'SAVING CHECKPOINT: {self.getCheckpointFile(i)}')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

    def cleanupPartialFiles(self, iteration):
        """Remove all partial files for the completed iteration"""
        folder = self.args.checkpoint
        partial_files = [f for f in os.listdir(folder) if f.startswith(f"partial_iter_{iteration}_")]
        
        for file in partial_files:
            try:
                os.remove(os.path.join(folder, file))
                log.info(f"Removed partial file {file}")
            except Exception as e:
                log.warning(f"Error removing partial file {file}: {e}")

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        log.info(f"Saving examples to {filename}")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True