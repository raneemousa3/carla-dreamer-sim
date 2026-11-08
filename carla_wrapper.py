from car_dreamer.carla_follow_env import CarlaFollowEnv
import numpy as np
import torch
import carla

def aggregate_dcts(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

class CarlaWrapper(CarlaFollowEnv):
    def __init__(self, config):
        super().__init__(config)
        self.collision_count = 0

    def get_obs(self, obs):
        """
        Extract image observation from the observation dictionary.
        """
        return {
            'visual': obs['camera'],
            'proprio': self.state()[0]
        }

    def get_distance(self):
        """
        Uses bounding boxes of vehicles to compute more accurate distance
        """
        def get_face_vertices(actor: carla.Actor, face: str) -> carla.Location:
            """
            face: 'front' or 'back'
            """
            bb = actor.bounding_box
            transform = actor.get_transform()
            x = bb.extent.x
            y = bb.extent.y
            z = bb.extent.z

            x_face = +x if face == 'front' else -x

            local_verts = [
                carla.Location(x=x_face, y=-y, z=-z),
                carla.Location(x=x_face, y= y, z=-z),
                carla.Location(x=x_face, y= y, z= z),
                carla.Location(x=x_face, y=-y, z= z),
            ]

            return [transform.transform(bb.location + v) for v in local_verts]
        
        def get_edges_from_verts(verts: list[carla.Location]) -> list[tuple[np.ndarray, np.ndarray]]:
            verts_np = [np.array([v.x, v.y, v.z]) for v in verts]
            return [
                (verts_np[0], verts_np[1]),
                (verts_np[1], verts_np[2]),
                (verts_np[2], verts_np[3]),
                (verts_np[3], verts_np[0]),
            ]
        
        def segment_distance(p1, p2, q1, q2):
            """Compute shortest distance between two 3D line segments [p1,p2] and [q1,q2]."""
            # From https://stackoverflow.com/a/52551983
            u = p2 - p1
            v = q2 - q1
            w = p1 - q1
            a = np.dot(u, u)
            b = np.dot(u, v)
            c = np.dot(v, v)
            d = np.dot(u, w)
            e = np.dot(v, w)

            D = a * c - b * b
            sc, sN, sD = 0.0, 0.0, D
            tc, tN, tD = 0.0, 0.0, D

            SMALL_NUM = 1e-8

            if D < SMALL_NUM:
                sN = 0.0
                sD = 1.0
                tN = e
                tD = c
            else:
                sN = (b * e - c * d)
                tN = (a * e - b * d)
                if sN < 0.0:
                    sN = 0.0
                    tN = e
                    tD = c
                elif sN > sD:
                    sN = sD
                    tN = e + b
                    tD = c

            if tN < 0.0:
                tN = 0.0
                if -d < 0.0:
                    sN = 0.0
                elif -d > a:
                    sN = sD
                else:
                    sN = -d
                    sD = a
            elif tN > tD:
                tN = tD
                if (-d + b) < 0.0:
                    sN = 0
                elif (-d + b) > a:
                    sN = sD
                else:
                    sN = (-d + b)
                    sD = a

            sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
            tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD

            dP = w + sc * u - tc * v
            return np.linalg.norm(dP)
        
        def min_edge_distance(edges1, edges2):
            min_dist = float("inf")
            for (p1, p2) in edges1:
                for (q1, q2) in edges2:
                    dist = segment_distance(p1, p2, q1, q2)
                    min_dist = min(min_dist, dist)
            return min_dist

        ego_face = get_face_vertices(self.ego, 'front')
        other_face = get_face_vertices(self.nonego, 'back')
        ego_edges = get_edges_from_verts(ego_face)
        other_edges = get_edges_from_verts(other_face)

        distance = min_edge_distance(ego_edges, other_edges)
        return distance




    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        _, init_state = self.reset()
        self.step([0, 0])
        _, goal_state = self.reset()
        return init_state, goal_state

    def reset(self, reset_to_state=None):
        """
        Reset the environment and return the initial observation.
        """
        self.reset_to_state = reset_to_state
        self.collision_count = 0
        obs = super().reset()
        obs = self.get_obs(obs)
        return obs, self.state()
    
    def update_env(self, env_info):
        """
        Update the environment with the given information.
        This method can be used to update the environment state or configuration.
        """
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Evaluate the current state against the goal state.
        This method can be used to determine how close the current state is to the goal state.
        """
        cur_ego_pos = np.array(cur_state[0][:2])
        cur_nonego_pos = np.array(cur_state[1][:2])
        cur = np.concatenate((cur_ego_pos, cur_nonego_pos))

        goal_ego_pos = np.array(goal_state[0][:2])
        goal_nonego_pos = np.array(goal_state[1][:2])
        goal = np.concatenate((goal_ego_pos, goal_nonego_pos))

        pos_diff = np.linalg.norm(cur - goal)

        success = pos_diff < 1.0  # Define success condition based on distance threshold
        state_dist = np.linalg.norm(np.array(cur_state) - np.array(goal_state))

        return {
            'success': success,
            'state_dist': state_dist
        }
    
    def prepare(self, seed, init_state):
        """
        Prepare the environment with a specific seed and initial state.
        This method can be used to set up the environment before starting an episode.
        """
        self.seed(seed)
        return self.reset(init_state)
    
    def step_multiple(self, actions):
        """
        Perform multiple steps in the environment with the given actions.
        This method can be used to execute a sequence of actions in the environment.
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dcts(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        # infos = aggregate_dcts(infos)
        return obses, rewards, dones, infos
    
    def rollout(self, seed, init_state, actions):
        """
        Perform a rollout in the environment with the given seed, initial state, and actions.
        This method can be used to execute a sequence of actions and collect observations, rewards, and other information.
        """
        obs, state = self.prepare(seed, init_state)
        
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
    
    def step(self, action):
        """
        Perform a single step in the environment with the given action.
        This method can be used to execute an action and return the next observation, reward, done status, and additional information.
        """
        obs, reward, done, info = super().step(action)
        obs = self.get_obs(obs)
        if self.is_collision():
            self.collision_count += 1
        return obs, reward, done, info
    
    def _is_terminal(self):
        terminal, info = super()._is_terminal()
        info.update({
            "collision_count": self.collision_count,
            "distance": self.get_distance(),
        })
        return terminal, info


