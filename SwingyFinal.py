
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # TESTING
        #print("Distance:")
        #print(state['tree']['dist'])


# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

# Some hyperparameters
binsize = 5#10
binsizeVel=4#5
max_height = 1000 # Not sure what the full dimensions are so I guessed until it stopped giving me errors!
max_height_tree=2000
max_width = 1000
max_vel = 40
# Need to play with these settings a lot
gamma = .98
epsilon = 0.0
# Set alpha - can do this more dynamically if desired
alpha = 1

a=3#4#5
b=3#4#5
c=2.5#3#3.5
lendist=5#5
lentree=5#5
lenvel=5#5  
#def Qsmooth(Q,cur_acc,cur_vel,cur_tree,cur_dist,cur_jump,cur_rew):
#    N_vel = Q.shape[1]/20  #normalization constant
#    N_tree = Q.shape[2]/20 #normalization constant
#    N_dist = Q.shape[3]/20 #normalization constant        
#    for ii in range(Q.shape[1]):
#        dist_metric_x=np.exp(-((((cur_vel)-(ii))/N_vel)**2))
#        if np.abs(cur_vel-ii)<5:
#            for jj in range(Q.shape[2]):
#                dist_metric_y=np.exp(-((((cur_tree)-(jj))/N_tree)**2))
#                if np.abs(cur_tree-jj)<5:
#                    for kk in range(Q.shape[3]):
#                        dist_metric_z=np.exp(-((((cur_dist)-(kk))/N_dist)**2))
#                        if np.abs(cur_dist-kk)<5:
#                            dist_metric=dist_metric_x*dist_metric_y*dist_metric_z
##                            print('dist metric')
##                            print(cur_rew)
#                            
#                            Q[cur_acc,ii,jj,kk,cur_jump]+=cur_rew*dist_metric
#    return Q

def Qsmooth(Q,cur_acc,cur_vel,cur_tree,cur_dist,cur_jump,cur_rew):
    N_vel = a*Q.shape[1]/20  #normalization constant
    N_tree = b*Q.shape[2]/20 #normalization constant
    N_dist = c*Q.shape[3]/20 #normalization constant     
    for ii in range(cur_vel-lenvel,cur_vel+lenvel+1):
        dist_metric_x=np.exp(-(((cur_vel-ii)/N_vel)**2))
        if ii>-1 and ii<Q.shape[1]:
            for jj in range(cur_tree-lentree,cur_tree+lentree+1):
                dist_metric_y=np.exp(-((((cur_tree)-(jj))/N_tree)**2))
                if jj>-1 and jj<Q.shape[2]:
                    for kk in range(cur_dist-lendist,cur_dist+lendist+1):#range(Q.shape[3]):
                        dist_metric_z=np.exp(-((((cur_dist)-(kk))/N_dist)**2))
                        if kk>-1 and kk<Q.shape[3]:
                            dist_metric=dist_metric_x*dist_metric_y*dist_metric_z
#                            print('dist metric')
#                            print(cur_rew)
                            
                            Q[cur_acc,ii,jj,kk,cur_jump]+=cur_rew*dist_metric
    return Q

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.accJ=None
        self.monVelJ=None
        self.treeJ=None
        self.distJ=None
        self.jumpLast=0
        self.epoch = 1
        self.max_reward = 0
        self.cumulative_reward = 0
        self.max_vel_meas=0
        self.count=0
        self.acc=0
        self.velSR=2
        self.velS=.25
        self.treeSR=3
        self.treeS=.25
        self.distSR=2
        self.distS=.25
        self.penalty=.9
        self.jumpCount=0
        self.jumpMean=0
        self.sinceLastJump=0
        # Q is an array with 6 dimensions:
        # 1) acceleration
        # 2) velocity
        # 3) monkey height minus tree bottom height (since gap is alway same, would be redundant to include top also)
        # 4) tree distance
        # 5) action
        self.Q = 0*np.ones([2, 2*int(max_vel/binsizeVel), int(max_height_tree/binsize), int(max_width/binsize), 2])
        self.Q[:,int(max_vel/binsizeVel):,int(.5*max_height_tree/binsize):,:,1]=-1*np.ones([2,int(max_vel/binsizeVel),int(.5*max_height_tree/binsize), int(max_width/binsize)])
        # N - counts number of actions taken in each state (same dimensions as Q)
        self.N = np.zeros([2, 2*int(max_vel/binsizeVel), int(max_height_tree/binsize), int(max_width/binsize), 2])
        
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.accJ=None
        self.monVelJ=None
        self.treeJ=None
        self.distJ=None
        self.cumulative_reward = 0
        self.epoch += 1
        self.count=0
        self.acc=0
        self.sinceLastJump=0
   
        
        
    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        
        # Pull in the current state in terms of the bins
        tree = int(np.floor(np.divide(state['monkey']['bot']-state['tree']['bot']+max_height,binsize)))
        monkey = int(np.floor(np.divide(state['monkey']['bot'],binsize)))
        monVel = int(np.floor(np.divide(state['monkey']['vel']+max_vel,binsizeVel)))
        monVelRaw=state['monkey']['vel']
#        print('monVelRaw')
#        print(monVelRaw)
        
        if np.abs(monVelRaw)>self.max_vel_meas:
            self.max_vel_meas=np.abs(monVelRaw)
#            print('Max monVelRaw')
#            print(self.max_vel_meas)
        if self.last_action:
            self.jumpMean=(self.jumpMean*self.jumpCount+monVelRaw)/(self.jumpCount+1)
            self.jumpCount=1
            
        
        # Special condition for distance (since it can be negative)
        # Give it max value if negative
        if state['tree']['dist'] > 0:
            dist = int(np.floor(np.divide(state['tree']['dist'],binsize)))
        else:
            dist = int(max_width/binsize-1)
        
        acc=0
        
        if not self.last_state == None:
            # Pull in the last state in terms of the bins
            last_tree = int(np.floor(np.divide(self.last_state['monkey']['bot']-self.last_state['tree']['bot']+max_height,binsize)))
            last_monkey = int(np.floor(np.divide(self.last_state['monkey']['bot'],binsize)))
            last_monVel = int(np.floor(np.divide(self.last_state['monkey']['vel']+max_vel,binsizeVel)))
            last_monVelRaw=self.last_state['monkey']['vel']
            
#            print('last_monVelRaw')
#            print(last_monVelRaw)
            
            bigAcc=(last_monVelRaw-monVelRaw)>2
            
            if bigAcc:
                acc=1
            else:
                acc=0
            # Special condition for distance (since it can be negative)
            # Give it max value if negative
            if state['tree']['dist'] > 0:
                last_dist = int(np.floor(np.divide(self.last_state['tree']['dist'],binsize)))
            else:
                last_dist = int(max_width/binsize-1)
        
            # Define a new action based on the Q values of different actions
            
            
            #Average with state you will jump to
            ####
            if acc>0:
                acc_raw=4
            else:
                acc_raw=1
            next_tree_raw=self.jumpMean+state['monkey']['bot']-state['tree']['bot']+max_height
            next_monVel_raw=self.jumpMean+max_vel-acc_raw#plus???
            if state['tree']['dist'] > 25:
                next_dist_raw = int(np.floor(np.divide(self.last_state['tree']['dist'],binsize)))-25
            else:
                next_dist_raw = 0
                
            next_monVel_bin=int(np.floor(np.divide(next_monVel_raw,binsizeVel)))
            next_tree_bin=int(np.floor(np.divide(next_tree_raw,binsize)))
            next_dist_bin=int(np.floor(np.divide(next_dist_raw,binsize)))    
            
            avgWithNext=1
            if next_monVel_bin<0 or next_monVel_bin>self.Q.shape[1]:
                avgWithNext=0
            if next_tree_bin<0 or next_tree_bin>self.Q.shape[2]:
                avgWithNext=0
            if next_dist_bin<0 or next_dist_bin>self.Q.shape[3]:
                avgWithNext=0
            
            ####
            
            #Average with state you will fall to
            ####
            if acc>0:
                acc_raw=4
            else:
                acc_raw=1
            fall_tree_raw=state['monkey']['vel']+state['monkey']['bot']-state['tree']['bot']+max_height
            fall_monVel_raw=state['monkey']['vel']+max_vel-acc_raw
            if state['tree']['dist'] > 25:
                fall_dist_raw = int(np.floor(np.divide(self.last_state['tree']['dist'],binsize)))-25
            else:
                fall_dist_raw = 0
                
            fall_monVel_bin=int(np.floor(np.divide(fall_monVel_raw,binsizeVel)))
            fall_tree_bin=int(np.floor(np.divide(fall_tree_raw,binsize)))
            fall_dist_bin=int(np.floor(np.divide(fall_dist_raw,binsize)))    
            
            avgWithFall=1
            if fall_monVel_bin<0 or fall_monVel_bin>self.Q.shape[1]:
                avgWithFall=0
            if fall_tree_bin<0 or fall_tree_bin>self.Q.shape[2]:
                avgWithFall=0
            if fall_dist_bin<0 or fall_dist_bin>self.Q.shape[3]:
                avgWithFall=0
            
            ####
            
            if avgWithNext:
                Qbest_next=np.max([self.Q[acc,next_monVel_bin,next_tree_bin,next_dist_bin,0],self.Q[acc,next_monVel_bin,next_tree_bin,next_dist_bin,1]])
                if avgWithFall:
                    Qbest_fall=np.max([self.Q[acc,fall_monVel_bin,fall_tree_bin,fall_dist_bin,0],self.Q[acc,fall_monVel_bin,fall_tree_bin,fall_dist_bin,1]])
                    if np.mean([self.Q[acc,monVel,tree,dist,1],Qbest_next])> np.mean([self.Q[acc,monVel,tree,dist,0],Qbest_fall]):
                    #if np.mean([self.Q[acc,monVel,tree,dist,1],self.Q[acc,next_monVel_bin,next_tree_bin,next_dist_bin,0]])> np.mean([self.Q[acc,monVel,tree,dist,0],self.Q[acc,fall_monVel_bin,fall_tree_bin,fall_dist_bin,0]]):
                        new_action = 1
                        print(self.Q[acc,monVel,tree,dist,1])
                    else:
                        new_action = 0                    
                else:
#                    if np.mean([self.Q[acc,monVel,tree,dist,1],self.Q[acc,next_monVel_bin,next_tree_bin,next_dist_bin,0]])> self.Q[acc,monVel,tree,dist,0]:
                    if np.mean([self.Q[acc,monVel,tree,dist,1],Qbest_next])> self.Q[acc,monVel,tree,dist,0]:
                        new_action = 1
                        print(self.Q[acc,monVel,tree,dist,1])
                    else:
                        new_action = 0
            else:
                if avgWithFall:
                    Qbest_fall=np.max([self.Q[acc,fall_monVel_bin,fall_tree_bin,fall_dist_bin,0],self.Q[acc,fall_monVel_bin,fall_tree_bin,fall_dist_bin,1]])
#                    if self.Q[acc,monVel,tree,dist,1] > np.mean([self.Q[acc,monVel,tree,dist,0],self.Q[acc,fall_monVel_bin,fall_tree_bin,fall_dist_bin,0]]):
                    if self.Q[acc,monVel,tree,dist,1] > np.mean([self.Q[acc,monVel,tree,dist,0],Qbest_fall]):
                        new_action = 1
                        print(self.Q[acc,monVel,tree,dist,1])
                    else:
                        new_action = 0
                else:
                    if self.Q[acc,monVel,tree,dist,1] > self.Q[acc,monVel,tree,dist,0]:
                        new_action = 1
                        print(self.Q[acc,monVel,tree,dist,1])
                    else:
                        new_action = 0
        
            # E-greedy replacement routine
            if self.N[acc,monVel,tree,dist, new_action] == 0:
                if npr.rand() < epsilon:
                    new_action = npr.rand() > 0.5
            else:
                if npr.rand() < np.divide(epsilon,(self.N[acc,monVel,tree,dist,new_action] + 1)**2): ### JIMMY EDIT: MADE IT EXPONENTIAL
                    new_action = npr.rand() > 0.5
            
            #don't ever fall off the bottom because that is stupid
            if state['monkey']['bot']<50 and acc==1:
                if state['monkey']['vel']<0:
                    new_action=1
                    print('auto jump')
            if state['monkey']['bot']<30 and acc==0:
                if state['monkey']['vel']<0:
                    new_action=1
                    print('auto jump')
                    
            #Dont jump off the top either
            if state['monkey']['bot']>800:     ## JIMMY EDIT: FIXED INEQUALITY
#                if state['monkey']['vel']>0:
                new_action=0
            # Get max Q result
            max_Q = np.max(self.Q[acc,monVel,tree,dist,:])
            print("max Q", max_Q)
            # Update Q
            #print(last_tree)
#            print(alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action]))
#            print(self.Q[acc,last_monVel, last_tree,last_dist,self.last_action])
            self.Q[acc,last_monVel, last_tree,last_dist,self.last_action] += alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#            cur_rew=alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#            cur_rew=alpha*(self.last_reward + gamma*max_Q)
#            self.Q=Qsmooth(self.Q,acc,last_monVel, last_tree,last_dist,self.last_action,cur_rew)
            
            ### Uncomment here to
            
            #This rewards when it last jumped for the reward it gets now
            if not self.accJ == None:
                if self.jumpLast!=1:
                    accJ=self.accJ
#                    print('accJ')
#                    print(accJ)
                    monVelJ=self.monVelJ
#                    print(monVelJ)
                    treeJ=self.treeJ
#                    print(treeJ)
                    distJ=self.distJ
#                    print(distJ)
                    actionJ=1
#                   print(self.last_reward)
#                   print(gamma*max_Q) 
                
#                    print(self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#                    self.Q[accJ,monVelJ,treeJ,distJ,actionJ]+=alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#                    print(self.Q[accJ,monVelJ, treeJ,distJ,actionJ])
#                    print(alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action]))
                    
                    
#                    if alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action]).size==0:
#                        valToAdd=0
#                    else:
#                        valToAdd=alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
                    if self.last_reward>0:
                        valToAdd=self.last_reward
                    else:
                        if state['monkey']['bot']-state['tree']['bot']<=0:
                            valToAdd=self.last_reward*(gamma**self.sinceLastJump)
                        else:
                            valToAdd=self.last_reward
                    
                    #self.Q[accJ,monVelJ, treeJ,distJ,actionJ] += valToAdd
                    cur_rew=valToAdd
                    
                    if cur_rew<0:
                        print('negative jump reward')
                    
                    self.Q=Qsmooth(self.Q,accJ,monVelJ, treeJ,distJ,actionJ,cur_rew)
            
            #positive rewards for all safe states
            if self.last_reward>0:
                last_tree_raw=self.last_state['monkey']['bot']-self.last_state['tree']['bot']+max_height
                last_monVel_raw=self.last_state['monkey']['vel']+max_vel
#                if state['tree']['dist'] > 60:
#                    last_dist_raw = 60
#                else:
#                    last_dist_raw = self.last_state['tree']['dist']
                last_dist_raw = 0
                
#                print('Raw parameters')
#                print(self.last_reward)
#                print(last_dist_raw)
#                print(last_monVel_raw)
#                print(last_tree_raw)
                
                if acc>0:
                    acc_raw=4
                else:
                    acc_raw=1
                
                inBounds=1
                countImproves=0
                while inBounds:
                    last_monVel_raw+=acc_raw#the velocity at the previous time step
                    last_tree_raw-=last_monVel_raw#the position at the previous time step, it used last time steps velocity to update position
                    last_dist_raw+=25#the tree always moves 25 forward
                    
#                    print('Raw parameters')
##                    print(self.last_reward)
#                    print(last_dist_raw)
#                    print(last_monVel_raw)
#                    print(last_tree_raw)
                        
                    last_monVel_bin=int(np.floor(np.divide(last_monVel_raw,binsizeVel)))
                    last_tree_bin=int(np.floor(np.divide(last_tree_raw,binsize)))
                    last_dist_bin=int(np.floor(np.divide(last_dist_raw,binsize)))
                    
                    #Make sure the velocity is in bounds
                    if last_monVel_bin<0:
                        inBounds=0
                        print('vel out of bounds')
                    if last_monVel_bin>=self.Q.shape[1]:
                        inBounds=0
                        print('vel out of bounds')
                                        
                    #Make sure the tree parameter is in bounds
                    if last_tree_bin<0:
                        inBounds=0
                        print('tree too low')
                        
                    if last_tree_bin>=self.Q.shape[2]:
                        inBounds=0
                        print('tree too high')
                    
                    #Make sure the dist is in bounds
                    if last_dist_bin<0:
                        inBounds=0
                        print('dist out of bounds')
                    if last_dist_bin>=self.Q.shape[3]:
                        inBounds=0
                        print('dist out of bounds')
                    
                    if inBounds:
                        #self.Q[acc,last_monVel_bin,last_tree_bin,last_dist_bin,0]+=self.last_reward
                        cur_rew=self.last_reward
                        self.Q=Qsmooth(self.Q,acc,last_monVel_bin,last_tree_bin,last_dist_bin,0,cur_rew)
                        countImproves+=1
                print('Improved')
                print(countImproves)
                
                #negative rewards for unsafe states
            if self.last_reward<0:
                last_tree_raw=self.last_state['monkey']['bot']-self.last_state['tree']['bot']+max_height
                last_monVel_raw=self.last_state['monkey']['vel']+max_vel
#                if state['tree']['dist'] > 60:
#                    last_dist_raw = 60
#                else:
#                    last_dist_raw = self.last_state['tree']['dist']
                last_dist_raw = 0
                
#                print('Raw parameters')
#                print(self.last_reward)
#                print(last_dist_raw)
#                print(last_monVel_raw)
#                print(last_tree_raw)
                
                if acc>0:
                    acc_raw=4
                else:
                    acc_raw=1
                
                inBounds=1
                countImproves=0
                cur_rew=self.last_reward
                while inBounds:
                    last_monVel_raw+=acc_raw#the velocity at the previous time step
                    last_tree_raw-=last_monVel_raw#the position at the previous time step, it used last time steps velocity to update position
                    last_dist_raw+=25#the tree always moves 25 forward
                    
#                    print('Raw parameters')
##                    print(self.last_reward)
#                    print(last_dist_raw)
#                    print(last_monVel_raw)
#                    print(last_tree_raw)
                        
                    last_monVel_bin=int(np.floor(np.divide(last_monVel_raw,binsizeVel)))
                    last_tree_bin=int(np.floor(np.divide(last_tree_raw,binsize)))
                    last_dist_bin=int(np.floor(np.divide(last_dist_raw,binsize)))
                    
                    #Make sure the velocity is in bounds
                    if last_monVel_bin<0:
                        inBounds=0
                        print('vel out of bounds')
                    if last_monVel_bin>=self.Q.shape[1]:
                        inBounds=0
                        print('vel out of bounds')
                                        
                    #Make sure the tree parameter is in bounds
                    if last_tree_bin<0:
                        inBounds=0
                        print('tree too low')
                        
                    if last_tree_bin>=self.Q.shape[2]:
                        inBounds=0
                        print('tree too high')
                    
                    #Make sure the dist is in bounds
                    if last_dist_bin<0:
                        inBounds=0
                        print('dist out of bounds')
                    if last_dist_bin>=self.Q.shape[3]:
                        inBounds=0
                        print('dist out of bounds')
                    
                    if inBounds:
                        #self.Q[acc,last_monVel_bin,last_tree_bin,last_dist_bin,0]+=self.last_reward
                        cur_rew=cur_rew*gamma#self.penalty
                        self.Q=Qsmooth(self.Q,acc,last_monVel_bin,last_tree_bin,last_dist_bin,0,cur_rew)
                        countImproves+=1
                print('Penalized')
                print(countImproves)
                
                # HERE
                
                
#smooth the Q along velocity
#            for i in range(1,self.velSR):
#                if last_monVel+i<self.Q.shape[1]:
#                    self.Q[acc,last_monVel+i, last_tree,last_dist,self.last_action] += (self.velS**i)*alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#                if last_monVel+i>-1:
#                    self.Q[acc,last_monVel-i, last_tree,last_dist,self.last_action] += (self.velS**i)*alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#            
#            #smooth the Q along tree
#            for j in range(1,self.treeSR):
#                if last_tree+j<self.Q.shape[2]:
#                    self.Q[acc,last_monVel, last_tree+j,last_dist,self.last_action] += (self.treeS**i)*alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#                if last_tree+i>-1:
#                    self.Q[acc,last_monVel, last_tree-j,last_dist,self.last_action] += (self.treeS**i)*alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#            
#            #smooth the Q along dist
#            for k in range(1,self.distSR):
#                if last_dist+k<self.Q.shape[3]:
#                    self.Q[acc,last_monVel, last_tree,last_dist+k,self.last_action] += (self.distS**i)*alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
#                if last_dist+k>-1:
#                    self.Q[acc,last_monVel, last_tree,last_dist-k,self.last_action] += (self.distS**i)*alpha*(self.last_reward + gamma*max_Q - self.Q[acc,last_monVel,last_tree,last_dist,self.last_action])
            
            
            # Update max_reward
            self.cumulative_reward += int(self.last_reward)
            if self.cumulative_reward > self.max_reward:
                self.max_reward = self.cumulative_reward
                
        
        # Do a random action to start off (ignoring the above Q decision)
        if self.last_state == None:
            new_action = npr.rand() > 0.99

        # Update N
        self.N[acc,monVel,tree,dist,new_action] += 1
        
        # Keeping track of reward   
        print("Max reward:", self.max_reward)
        
        # Reset actions for next iteration and submit new action
        self.last_action = new_action
        self.last_state  = state
        if new_action==1:
#            self.last_jump_params=[acc,last_monVel, last_tree,last_dist,self.last_action]
            self.accJ=acc
            self.monVelJ=monVel
            self.treeJ=tree
            self.distJ=dist
            self.jumpLast=1
            self.sinceLastJump=0
        else:
            self.jumpLast=0
            self.sinceLastJump+=1
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 500, 10)

	# Save history. 
	np.save('hist',np.array(hist))