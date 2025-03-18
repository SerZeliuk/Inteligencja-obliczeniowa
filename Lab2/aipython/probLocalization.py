# probLocalization.py - Controlled HMM and Localization example
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from probHMM import HMMVEfilter, HMM
from display import Displayable
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

class HMM_Controlled(HMM):
    """A controlled HMM, where the transition probability depends on the action.
       Instead of the transition probability, it has a function act2trans
       from action to transition probability.
       Any algorithms need to select the transition probability according to the action.
    """
    def __init__(self, states, obsvars, pobs, act2trans, indist):
        self.act2trans = act2trans
        HMM.__init__(self, states, obsvars, pobs, None, indist)


local_states = list(range(16))
door_positions = {2,4,7,11}
def prob_door(loc): return 0.8 if loc in door_positions else 0.1
local_obs = {'door':[prob_door(i) for i in range(16)]}
act2trans = {'right': [[0.1 if next == current
                        else 0.8 if next == (current+1)%16
                        else 0.074 if next == (current+2)%16
                        else 0.002 for next in range(16)]
                           for current in range(16)],
             'left': [[0.1 if next == current
                        else 0.8 if next == (current-1)%16
                        else 0.074 if next == (current-2)%16
                        else 0.002 for next in range(16)]
                          for current in range(16)]}
hmm_16pos = HMM_Controlled(local_states, {'door'}, local_obs,
                               act2trans, [1/16 for i in range(16)])
class HMM_Local(HMMVEfilter):
    """VE filter for controlled HMMs
    """
    def __init__(self, hmm):
        HMMVEfilter.__init__(self, hmm)

    def go(self, action):
        self.hmm.trans = self.hmm.act2trans[action]
        self.advance()

loc_filt = HMM_Local(hmm_16pos)
# loc_filt.observe({'door':True}); loc_filt.go("right"); loc_filt.observe({'door':False}); loc_filt.go("right");  loc_filt.observe({'door':True})
# loc_filt.state_dist

class Show_Localization(Displayable):
    def __init__(self, hmm, fontsize=10):
        self.hmm = hmm
        self.fontsize = fontsize
        self.loc_filt = HMM_Local(hmm)
        fig,(self.ax) = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ## Set up buttons:
        left_butt = Button(plt.axes([0.05,0.02,0.1,0.05]), "left")
        left_butt.label.set_fontsize(self.fontsize)
        left_butt.on_clicked(self.left)
        right_butt = Button(plt.axes([0.25,0.02,0.1,0.05]), "right")
        right_butt.label.set_fontsize(self.fontsize)
        right_butt.on_clicked(self.right)
        door_butt = Button(plt.axes([0.45,0.02,0.1,0.05]), "door")
        door_butt.label.set_fontsize(self.fontsize)
        door_butt.on_clicked(self.door)
        nodoor_butt = Button(plt.axes([0.65,0.02,0.1,0.05]), "no door")
        nodoor_butt.label.set_fontsize(self.fontsize)
        nodoor_butt.on_clicked(self.nodoor)
        reset_butt = Button(plt.axes([0.85,0.02,0.1,0.05]), "reset")
        reset_butt.label.set_fontsize(self.fontsize)
        reset_butt.on_clicked(self.reset)
        ## draw the distribution
        plt.subplot(1, 1, 1)
        self.draw_dist()
        plt.show()

    def draw_dist(self):
        self.ax.clear()
        plt.ylim(0,1)
        plt.ylabel("Probability", fontsize=self.fontsize)
        plt.xlabel("Location", fontsize=self.fontsize)
        plt.title("Location Probability Distribution", fontsize=self.fontsize)
        plt.xticks(self.hmm.states,fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        vals = [self.loc_filt.state_dist[i] for i in self.hmm.states]
        self.bars = self.ax.bar(self.hmm.states, vals, color='black')
        self.ax.bar_label(self.bars,["{v:.2f}".format(v=v) for v in vals], padding = 1, fontsize=self.fontsize)
        plt.draw()

    def left(self,event):
        self.loc_filt.go("left")
        self.draw_dist()
    def right(self,event):
        self.loc_filt.go("right")
        self.draw_dist()
    def door(self,event):
        self.loc_filt.observe({'door':True})
        self.draw_dist()
    def nodoor(self,event):
        self.loc_filt.observe({'door':False})
        self.draw_dist()
    def reset(self,event):
        self.loc_filt.state_dist = {i:1/16 for i in range(16)}
        self.draw_dist()

# Show_Localization(hmm_16pos)
# Show_Localization(hmm_16pos, fontsize=15) # for demos - enlarge window

if __name__ == "__main__":
    print("Try: Show_Localization(hmm_16pos)")

