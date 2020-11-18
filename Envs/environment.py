import gym
from gym import spaces
import numpy as np

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, state_callback=None,
                 info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.state_callback = state_callback # Flora Fu @20200728
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            self.action_space.append(spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32))
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info = dict()
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        ents, caps = self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            # info_n['n'].append(self._get_info(agent))

        for i, (ent, cap) in enumerate(zip(ents, caps)):
            # print('-----------Checking I'+str(i), '-----------')
            temp = {'dcap': None, 'tcap': np.inf, 'tent': np.inf}
            event = False # either capture or enter
            if ent is True:
                # print('entered')
                temp['tent'] = self.world.t
                event = True
            if cap:
                # print('captured')
                temp['dcap'] = 'D' + str(cap[0])
                temp['tcap'] = self.world.t
                event = True
            if event:
                info.update({'I'+str(i):temp})

        # print(self.world.t, info)

        return obs_n, reward_n, done_n, info

    def reset(self, *arg, **kwarg):
        # print(kwarg)
        self.reset_callback(self.world, *arg, **kwarg)
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n 

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def get_state(self):
        if self.state_callback is None:
            return np.zeros(0)
        return self.state_callback(self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        agent.action.u = action
        agent.action.c = np.zeros(self.world.dim_c)

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                from Envs import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            from Envs import rendering
            self.render_geoms = {}
            self.render_geoms_xform = []
            for entity in self.world.entities:
                res = 3 if 'agentI' in entity.name else 30
                body = rendering.make_circle(entity.size, res=res)
                geom = {entity.name+':body': body}
                xform = rendering.Transform()
                if 'agent' in entity.name: # same color as the defender that is assigned to the intruder @FloraFu 20200805
                    color = entity.color
                    if 'I' in entity.name:
                        for d in self.world.defenders:
                            if entity.id in d.state.o: 
                                color=d.color
                                break
                    body.set_color(*color)
                    if entity.r is not None: # capture disk @FloraFu 20200805
                        cdisk = rendering.make_circle(entity.r, filled=False) 
                        cdisk.set_color(*entity.color)
                        geom.update({entity.name+':cdisk': cdisk})
                        # print('added capture disk')
                    if 'agentD' in entity.name and entity.Ri is not None: # sensing disk @FloraFu 20200805
                        sdisk = rendering.make_circle(entity.Ri) 
                        sdisk.set_color(*entity.color, alpha=0.1)
                        geom.update({entity.name+':sdisk': sdisk})
                else:
                    # if 'landmark 0' in entity.name:
                    if entity.size >0.255:
                        body.set_color(*entity.color, alpha=0.25)
                    else:
                        body.set_color(*entity.color)
                for key, g in geom.items():
                    # xform = rendering.Transform()
                    g.add_attr(xform)
                self.render_geoms.update(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for key, geom in self.render_geoms.items():
                    viewer.add_geom(geom)

        else: # @FloraFu 20200805 use the color of the defender that the intruder is assigned to, otherwise use original color
            from Envs import rendering
            for intruder in self.world.intruders:
                if not intruder.state.a: 
                    self.render_geoms[intruder.name+':body'].set_color(1., 0.1, 0.1)
                    continue
                for d in self.world.defenders:
                    if intruder.id in d.state.o: 
                        self.render_geoms[intruder.name+':body'].set_color(*d.color)
                        break
                    self.render_geoms[intruder.name+':body'].set_color(*intruder.color) 
                
        results = []
        for i in range(len(self.viewers)):
            mode = 'rgb_array'; # Added by Qingrui Zhang
            # from multiagent import rendering
            from Envs import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                cam_range *= 6
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            self.viewers[i].render(return_rgb_array = mode=='rgb_array') # Added by Qingrui Zhang@20200117
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
    
    # Newly added by Qingrui Zhang @20200117
    def close(self):
        for i in range(len(self.viewers)):
            # close viewers (if necessary)
            if self.viewers[i] is not None:
                self.viewers[i].close();
                self.viewers[i]=None;
        self._reset_render();


