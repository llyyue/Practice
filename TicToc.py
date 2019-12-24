# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:45:13 2019

@author: llyyue
"""
import numpy as np

class Environment:
    def __init__(self, sizeX, sizeY, winLength):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.winLength = winLength
#        self.board = np.zeros((self.sizeX, self.sizeY))
        self.board = np.array([[1,1,0],[0,1,0],[0,0,1]])
        self.x = -1 # represents an x on the board, player 1
        self.o = 1 # represents an o on the board, player 2
        self.winner = None
        self.ended = False
        self.num_states = 3**(self.sizeX * self.sizeY)
    
    def is_empty(self, i, j):
        return self.board[i,j] == 0
    
    def is_draw(self):
        return self.ended and self.winner is None

    def reward(self, player):
        # no reward until game is over
        if not self.game_over():
            return 0
        return 1 if self.winner == player else 0
      
    def get_state(self):
        sum=0
        for i in range(sizeX):
            for j in range(sizeY):
                sum= sum*3 + (self.board[i,j]+1)
        return sum
        
    def game_over(self):
        if self.ended:
            return True
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                p = self.board[i,j]
                bFinished = False
                
                if (i+self.winLength<=self.sizeX):
                    bFinished = True
                    for k in range( self.winLength):
                        if(p* self.board[i+k,j]!=1):
                            bFinished = False
                            break
                        
                if (not bFinished) and (j+self.winLength<=self.sizeY):
                    bFinished = True
                    for k in range(self.winLength):
                        if(p* self.board[i,j+k]!=1):
                            bFinished = False
                            break
                
                if (not bFinished) and (i+self.winLength<=self.sizeX) and (j+self.winLength<=self.sizeY):
                    bFinished = True
                    for k in range(self.winLength):
                        if(p* self.board[i+k,j+k]!=1):
                            bFinished = False
                            break
                
                if bFinished:
                    self.winner= p
                    self.ended = True
        
        if np.all((self.board == 0) == False):
            self.winner= None
            self.ended = True
        
        return self.ended
    
    def display_board(self):
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                if(self.board[i,j]==self.o):
                    print('O',end="")
                elif(self.board[i,j]==self.x):
                    print('X',end="")
                else:
                    print(' ',end="")
                print('|',end="") 
            print("")
            for j in range(self.sizeY):
                print('--',end="")
            print("")
                
            
class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps # probability of choosing random action instead of greedy
        self.alpha = alpha # learning rate
        self.verbose = False
        self.state_history = []



if __name__ == '__main__':
    sizeX=3
    sizeY=3
    env = Environment(sizeX, sizeY,3)
    t=env.game_over()
    env.display_board()