import numpy as np
from math import sqrt

def norm(x):
    return sqrt(sum([xx**2 for xx in x]))

def dist(x, y):
    return norm(x-y)

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # active
        self.a = None
        # communication utterance
        self.c = None
        # number of capture/being captured (Flora Fu @20200715)
        self.n = None
        # whether enter the target, currently only applies for intruder (Flora Fu @20200716)
        self.e = None
        # the current opponent assigned, currently only applies for defenders (Flora Fu @20200716)
        self.o = []
        # efficiency of the intruders assigned, currently only applies for defenders
        self.f = []
        # assignment algorithm used
        self.s = None
        # time of capture and enter
        self.te = None
        self.tc = None

class AgentMem(object):
    """docstring for AgentMem"""
    def __init__(self):
        self.n = None
        self.e = None
        self.init_p_pos = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.00
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self, ind=0):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # self.u_noise = 0.5
        # self.u_noise = 100
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # self.u_range = 1.25
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # memory
        self.mem = AgentMem() # Flora Fu @20200721
        # script behavior to execute
        self.action_callback = None

        #agent label; added by Qingrui Zhang@20190119;
        self.id = ind

        self.r = None # capture range, Flora Fu @20200721
        self.Rd = 5. # sensing range, Flora Fu @20200805
        self.Ri = 5. # sensing range, Flora Fu @20200805

        self.neigh_i = []
        self.neigh_d = [] # neighbours within sensing range, Flora Fu @20200805

        # print('from agent', self.size)
        

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 2
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # current time
        self.t = 0.

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        for agent in self.agents:
            agent.state.p_vel = agent.action.u
            agent.state.p_pos += agent.state.p_vel * self.dt
            # print(agent.name, norm(agent.state.p_vel), agent.state.p_pos)

        self.t += self.dt # FloraFu @20200806   

class Game(World): # @FloraFu 20200729
    """docstring for Game"""
    def __init__(self):
        super(Game, self).__init__()

    @property
    def defenders(self):
        return [agent for agent in self.agents if 'D' in agent.name]

    @property
    def intruders(self):
        return [agent for agent in self.agents if 'I' in agent.name]

    def set_iselect_mode(self, mode):
        for d in self.defenders:
            d.iselect_mode = mode

    def update_world_state(self):
        # update defenders' memories
        for defender in self.defenders:
            defender.mem.n = [i for i in defender.state.n]
        
        # update intruders' memories AND current state
        ents, caps = [], []
        # print('before cheking:', self.intruders[-1].name, self.intruders[-1].state.a)
        for intruder in self.intruders:
            intruder.mem.e = intruder.state.e
            intruder.mem.n = [d for d in intruder.state.n]
            ent, dlist = None, None

            if intruder.state.a:
                # print('checking', intruder.name)
                ent = intruder.enter_callback(intruder, self)
                if ent:
                    # print(intruder.name, 'enters')
                    intruder.state.e = True
                    intruder.state.a = False
                    intruder.state.te = self.t
                dlist = intruder.capture_callback(intruder, self)
                if dlist: 
                    # print(intruder.name, 'captured')
                    intruder.state.n = dlist
                    intruder.state.a = False
                    intruder.state.tc = self.t
                    for did in dlist:
                        self.defenders[did].state.n.append(intruder.id)
                # if norm(intruder.state.p_vel) < 1e-6:
                #     intruder.state.a = False
            ents.append(ent)
            caps.append(dlist)
        # print('after cheking:', self.intruders[-1].name, self.intruders[-1].state.a)
        
        # update neighbours
        for defender in self.defenders:
            self.update_neighbours_defender(defender)
        for intruder in self.intruders:
            self.update_neighbours_intruder(intruder)

        return ents, caps

    def update_neighbours_defender(self, agent):
        # print('!!!!!!', dist(np.array([3., 4.]), np.array([0, .2])))
        # print([dist(other.state.p_pos, agent.state.p_pos) for other in self.intruders])
        agent.neigh_i = [other.id for other in self.intruders if other.state.a and dist(other.state.p_pos, agent.state.p_pos) < agent.Ri]
        agent.neigh_d = [other.id for other in self.defenders if other is not agent and dist(other.state.p_pos, agent.state.p_pos) < agent.Rd]

    def update_neighbours_intruder(self, agent):
        agent.neigh_i = [other.id for other in self.intruders if other is not agent and dist(other.state.p_pos, agent.state.p_pos) < agent.Ri]
        agent.neigh_d = [other.id for other in self.defenders if dist(other.state.p_pos, agent.state.p_pos) < agent.Rd]

    def step(self):
        super(Game, self).step()
        ents, caps = self.update_world_state()
        return ents, caps

                        