import numpy as np
import matplotlib.pyplot as plt

import discord
import re


class battleplans:
    def __init__ (self):
        """ initialize a zonotope of dimension dim with a line of values """
        self.ignore_skulls        = False
        self.ignore_single_shields= False
        self.ignore_double_shields= False
        self.ignore_hollow_swords = False
        self.ignore_double_swords = False
        self.double_attack_roll   = False
        self.cannot_sacrifice     = False

        self.graph                = False

    def read (self, regex):
        if (re.search(r".*skull.*"            , regex)!=None):
            self.ignore_skulls        = True
        if (re.search(r".*single.shield.*"     , regex)!=None):
            self.ignore_single_shields= True
        if (re.search(r".*double.shield.*"     , regex)!=None):
            self.ignore_double_shields= True
        if((re.search(r".*hollow.sword.*"      , regex)!=None)or(re.search(r".*half.sword.*", regex)!=None)):
            self.ignore_hollow_swords = True
        if (re.search(r".*double.sword.*"      , regex)!=None):
            self.ignore_double_swords = True
        if((re.search(r".*double.attack.roll*" , regex)!=None)or(re.search(r".*double.roll*", regex)!=None)):
            self.double_attack_roll   = True
        if (re.search(r".*sacrifice.*"         , regex)!=None):
            self.cannot_sacrifice     = True

        if (re.search(r".*graph.*"             , regex)!=None):
            self.graph                = True

    def tostring (self):
        response =''
        ignore = 0
        if (self.ignore_skulls        ):
            response+= (ignore==0)*" Ignoring " + (ignore>0)*", " + "skulls"        ; ignore+=1
        if (self.ignore_single_shields):
            response+= (ignore==0)*" Ignoring " + (ignore>0)*", " + "single shields"; ignore+=1
        if (self.ignore_double_shields):
            response+= (ignore==0)*" Ignoring " + (ignore>0)*", " + "double shields"; ignore+=1
        if (self.ignore_hollow_swords ):
            response+= (ignore==0)*" Ignoring " + (ignore>0)*", " + "hollow swords" ; ignore+=1
        if (self.ignore_double_swords ):
            response+= (ignore==0)*" Ignoring " + (ignore>0)*", " + "double swords" ; ignore+=1
        response += (ignore>0)*"."
        if (self.double_attack_roll   ):
            response+= " Doubling the attack roll."
        if (self.cannot_sacrifice     ):
            response+= " Attacker cannot sacrifice warbands."
        return (response)


def factorialLog (n):
    # returns an array contening ln(0!) to ln(n!)
    # It is a way of avoiding numerical errors from multiplying very small numbers with very large number
    factorial_log = np.zeros (n+1)
    for i in range (2, n+1):
        factorial_log[i] = factorial_log [i-1] + np.log (i)
    return (factorial_log )
    



def attackDices (n, bp):
    # returns a 2n+1 * n array, cell value is the probability of the event
    # line   i : number of swords
    # column j : number of skulls
    attack_probas = np.zeros ((2*n+1, n+1))

    fct = factorialLog (n)
    # for each dice face, enumerate all possible total
    for k in range (n+1):
        # k = number of plain swords
        for j in range (n+1-k):
            # j = number of skulls (correspondance with the result array)
            l = n - k - j # l = number of half swords
            i = (bp.ignore_double_swords==False)*2*j + k + (bp.ignore_hollow_swords==False)*l//2 # i = total number of swords (correspondance with the result array)
            # P =      exp( ln(n!) - ln(l!) - ln(k!) - ln(j!) - l ln(2) - k ln(3) - j ln(6) ) binomial law, using exp and log to reduce numerical errors
            proba = np.exp( fct[n] - fct[l] - fct[k] - fct[j] - l*np.log(2) - k*np.log(3) - j*np.log(6) )
            attack_probas[i, (bp.ignore_skulls==False)*j]+= proba

    #print (np.sum (attack_probas)) #verifies if the sum of all cells is 1
    return (attack_probas)


def defensDices (n, bp):
    # returns a 2n+1 * n array, cell value is the probability of the event
    # line   i : number of shields
    # column j : number of x2
    defens_probas = np.zeros ((2*n+1, n+1))

    fct = factorialLog (n)

    #for each dice face, enumerate all possible total
    for j in range (n+1):
        # j = number of x2 (correspondance with the result array)
        for k in range (n+1-j):
            # k = number of single shields
            for l in range (n+1-j-k):
                # l = number of double shields
                m = n - j - k - l # m = number of empty faces
                i = (bp.ignore_single_shields==False)*k + (bp.ignore_double_shields==False)*2*l # i = total number of shields (correspondance with the result array)
                # P =      exp( ln(n!) - ln(j!) - ln(k!) - ln(l!) - ln(m!) - j ln(6) - k ln(3) - l ln(6) - m ln(3)) binomial law, using exp and log to reduce numerical errors
                proba = np.exp( fct[n] - fct[j] - fct[k] - fct[l] - fct[m] - j*np.log(6) - k*np.log(3) - l*np.log(6) - m*np.log(3) )
                defens_probas[i, j]+= proba
    #print (np.sum (defens_probas)) #verifies if the sum of all cells is 1
    return (defens_probas)

def defensLaw (n):
    # computes the law and dispalys it
    defens_law = np.zeros (50+1) #no use to go beyond 50
    defens_probas = defensDices (n, bp)
    for i in range (2*n+1):
        for j in range (n+1):
            k = min(50, i*(2**j))
            defens_law[k]+=defens_probas[i, j]
    return (defens_law)
        
def cumulative (law, increasing=True):
    n = len(law)
    cumul_law = np.zeros (n)
    if (increasing):
        cumul_law[0] = law[0]
        for i in range(1, n):
            cumul_law[i] = cumul_law[i-1] +law[i]
    else:
        cumul_law[n-1] = law[n-1]
        for i in range(n-2, -1, -1):
            cumul_law[i] = cumul_law[i+1] +law[i]
    return (cumul_law)

def campaignOdds (attack_dices, attack_warbands, defens_dices, defens_warbands, bp=battleplans()):
    # computes the proba of each dices separately
    attack_probas = attackDices (attack_dices, bp)
    defens_probas = defensDices (defens_dices, bp)

    # result arrays
    victor_survivors = np.zeros (attack_warbands+1)
    defeat_survivors = np.zeros (attack_warbands+1)

    # range the 2 dimensions of each vector (highest complexity)
    for jatt in range (attack_dices+1):
        # ja = number of skulls
        current_warbands = max(attack_warbands - jatt, 0) # if skulls are ignored, does not remove warbands. the max is their because attacker can't have negative warbands
        for jdef in range (defens_dices+1):
            # jd = number of x2
            multiplicator = 2**jdef
            for idef in range (2*defens_dices+1):
                # id = base number of shields
                shields = idef*multiplicator
                for iatt in range (2*attack_dices+1):
                    # ia = number of swords
                    differential = max(1 + shields + defens_warbands - (1 + (bp.double_attack_roll))*iatt, 0) #max because if negative the attacker wins automatically
                    if ((bp.cannot_sacrifice==False)*current_warbands >= differential):
                        # campaign is a success
                        final_warbands = current_warbands-differential #sacrifice warbands to reach target
                        victor_survivors[final_warbands]+= attack_probas[iatt, jatt]*defens_probas[idef, jdef] # add probability
                    else :
                        #campaign is lost
                        final_warbands = current_warbands//2+current_warbands%2 #lose half warbands rounded down
                        defeat_survivors[final_warbands]+= attack_probas[iatt, jatt]*defens_probas[idef, jdef]

    victor_chance = np.sum (victor_survivors)
    defeat_chance = np.sum (defeat_survivors)

    if (bp.graph):
        plt.figure (1)
        plt.bar (range(attack_warbands+1), cumulative(victor_survivors, increasing=False), color="blue")
        plt.xlabel ("remaining warbands")
        plt.ylabel ("probability")
        plt.savefig ('warbands.jpg', bbox_inches = 'tight')

    return (victor_chance, victor_survivors, defeat_survivors)

"""
plt.figure (2)
plt.bar (range(attack_warbands+1), cumulative(defeat_survivors, increasing=False), color="red")
plt.xlabel ("remaining warbands")
plt.ylabel ("probability")
plt.title ("defeat chance "+ str(defeat_chance))
plt.show()
"""
    
def defensChart ():
    #Computes median and quartiles for the defens law
    max_dice = 12 #the maximum of the chart
    quart1 = np.zeros (max_dice) #25%
    median = np.zeros (max_dice) #50%
    quart3 = np.zeros (max_dice) #75%
    for n in range (1, max_dice+1):
        cumul_law = cumulative(defensLaw(n))
        for i in range (50):
            if (cumul_law[i]<=0.25)and(cumul_law[i+1]>0.25):
                quart1[n-1] = i 
            if (cumul_law[i]<=0.50)and(cumul_law[i+1]>0.50):
                median[n-1] = i 
            if (cumul_law[i]<=0.75)and(cumul_law[i+1]>0.75):
                quart3[n-1] = i
    plt.figure(10)
    plt.bar (range(1, max_dice+1), quart3, color="purple")
    plt.bar (range(1, max_dice+1), median, color="blue"  )
    plt.bar (range(1, max_dice+1), quart1, color="green" )
    plt.legend(["75%", "50%", "25%"] )
    plt.xlabel ("defence dice")
    plt.ylabel ("shields")
    


    


#plt.figure (3)
#plt.bar (range (51), cumulative(defensLaw(10)), color="blue")

#defensChart()


#campaignOdds (3, 3, 1, 1)
#campaignOdds (20, 20, 10, 10)



###################
### DISCORD BOT ###
###################
TOKEN = "ODUyOTI3MDE0ODY4MzUzMDc0.YMN8Lg.MQVubPgQ3ys509MU7rnPI4Makz0"

client = discord.Client()

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    


    if message.author == client.user:
        return

    if (message.content).startswith('-campaign '):
        regex = re.search(r"^-campaign (\d+) (\d+) (\d+) (\d+)?(.*)", message.content)
        if (regex==None):
            response = "Gnn?\nPlease start your request by -campaign number number number number"
            await message.channel.send(response)
            return
        elif ((regex[1]==None)or(regex[2]==None)or(regex[3]==None)or(regex[4]==None)):
            response = "Gnn?\nPlease start your request by -campaign number number number number"
            await message.channel.send(response)
            return


        bp = battleplans ()
        if (regex[5]!=None):
            bp.read(regex[5])
        
        
        #compute odds
        victor_chance, victor_survivors, defeat_survivors = campaignOdds (int(regex[1]), int(regex[2]), int(regex[3]), int(regex[4]), bp)

        #message
        response = "Campaign with "
        response+= regex[1] +  " attack dice, "
        response+= regex[2] +  " attacking warbands, "
        response+= regex[3] +  " defence dice, "
        response+= regex[4] +  " defending warbands."

        response+= bp.tostring()
        
        
        response+= "\nVictory chance: " + str(int(victor_chance*100+0.5)) + "%"
        await message.channel.send(response)

        if (bp.graph):
            await message.channel.send(file=discord.File('warbands.jpg'))

client.run(TOKEN)



