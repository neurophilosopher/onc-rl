# from OpenGL import GLU
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import random as rng
from PIL import Image,ImageDraw
import datetime
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYBNN_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "pybnn", "bin"))
if PYBNN_DIR not in sys.path:
    sys.path.insert(0, PYBNN_DIR)

import pybnn

class TWsearchEnv:
    def __init__(self,env,filter_len, mean_len, record_video=None, partial_obs=False):
        self.env = env
        self.filter_len = filter_len
        self.mean_len=mean_len
        self.record_video = record_video
        self.video_writer = None
        self.partial_obs = partial_obs

    def TensorRGBToImage(self,tensor):
        new_im = Image.new("RGB",(tensor.shape[1],tensor.shape[0]))
        pixels=[]
        for y in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                r = tensor[y][x][0]
                g = tensor[y][x][1]
                b = tensor[y][x][2]
                pixels.append((r,g,b))
        new_im.putdata(pixels)
        return new_im

    def input_size(self):
        if self.partial_obs:
            return 8
        return int(self.env.observation_space.shape[0])

    def output_size(self):
        return int(self.env.action_space.shape[0])

    def get_observation_slice(self, obs):
        if self.partial_obs:
            return obs[:8]
        return obs

    def set_observations_for_lif(self,obs,observations):
        obs_slice = self.get_observation_slice(obs)
        v = np.dot(obs_slice,self.w_in)

        observations[0] = float(v[0])
        observations[1] = float(obs[1])

    def post_process_action(self,action):
        action = np.array(action)
        actions = np.dot(action,self.w_out)
        return actions

    def render_frame(self):
        frame = self.env.render()
        renderer = getattr(self.env.unwrapped, "mujoco_renderer", None)
        if renderer is not None:
            viewer = renderer._viewers.get(self.env.render_mode)
            if viewer is not None:
                torso_id = self.env.unwrapped.model.body("torso").id
                torso_pos = self.env.unwrapped.data.xpos[torso_id]
                cam = viewer.cam
                cam.lookat[0] = float(torso_pos[0])
                cam.lookat[1] = float(torso_pos[1])
                cam.lookat[2] = 0.6
                cam.distance = 5.0
                cam.azimuth = 120.0
                cam.elevation = -12.0
                frame = self.env.render()
        return frame

    def run_one_episode(self,do_render=False):
        total_reward = 0
        obs, _ = self.env.reset()
        self.lif.Reset()
        if(do_render):
            rewardlog = open('rewardlog.log','w')
            self.lif.DumpClear('lif-dump.csv')
            if self.record_video and self.video_writer is None:
                self.video_writer = imageio.get_writer(self.record_video, fps=60)

        observations = []
        for i in range(0,2):
            observations.append(float(0))

        self.set_observations_for_lif(obs,observations)
        actions = np.zeros(1)
        self.lif.Update(observations,0.01,10)

        total_reward=np.zeros(1)
        gamma = 1.0
        time =0.0

        start_pos=0
        has_started=False
        i=0

        done2 = False
        while 1:
            action = self.lif.Update(observations,0.01,10)
            actions = self.post_process_action(action)
            if(do_render):
                print('T/R: '+str(time)+', '+str(total_reward)+': '+str(obs[1])+', '+str(obs[4])+', '+str(np.arcsin(float(obs[3]))))
            obs, r, terminated, truncated, info = self.env.step(actions)
            self.set_observations_for_lif(obs,observations)

            # if(not done2 and do_render):
            #     done2 = np.abs(np.arcsin(float(obs[3]))) > 0.2
            # if(not done2):
            #     max_bonus = 200.0/1000.0
            #     bonus = (1.0-abs(float(obs[0])))*max_bonus
            #     if(r>0.0):
            #         total_reward+=bonus*gamma

            total_reward += r*gamma
                #gamma = gamma*gamma
            time += 0.0165

            if(do_render):
                rewardlog.write(str(total_reward)+'\n')
                rewardlog.flush()
                self.lif.DumpState('lif-dump.csv')
                frame = self.render_frame()
                if self.video_writer is not None:
                    self.video_writer.append_data(frame)
                # screen = env.render(mode='rgb_array')
                # print('Img shape: '+str(screen.shape))
                # pic = TensorRGBToImage(screen)
                # pic.save('vid/img_'+str(i).zfill(5)+'.png')
                # phi = np.arcsin(obs[3])
                # print('Obs: '+str(phi)+', '+str(obs[4])+' Act: '+str(actions[0]))

                if(time >= 16.5):
                    return
            elif(terminated or truncated):
                break
            i+=1
        # print('Return: '+str(total_reward))
        return np.sum(total_reward)

    def evaluate_avg(self):
        N = 50

        returns = np.zeros(N)
        for i in range(0,N):
            returns[i]= self.run_one_episode()

        return np.mean(returns)


    def run_multiple_episodes(self):
        returns = np.zeros(self.filter_len)
        for i in range(0,self.filter_len):
            returns[i]= self.run_one_episode()

        sort = np.sort(returns)
        worst_cases = sort[0:self.mean_len]

        return [np.mean(worst_cases),np.mean(returns)]

    def load_tw(self,filename):
        self.lif = pybnn.LifNet(filename)
        #lif.AddBiSensoryNeuron(1,6,-0.3,0.3)
        # lif.AddBiSensoryNeuron(1,6,-0.03,0.03)
        # lif.AddBiSensoryNeuron(7,0,-0.15,0.15)


        # lif.AddBiSensoryNeuron(1,6,-0.03,0.03)
        self.lif.AddBiSensoryNeuron(1,6,-1,1)
        self.lif.AddBiSensoryNeuron(7,0,-1,1)

        self.lif.AddMotorNeuron(9,1)
        self.lif.AddMotorNeuron(10,1)

        # self.lif.AddBiMotorNeuron(9,10,-1,1)

        self.lif.Reset()

        aux_files = filename.replace(".bnn",".npz")
        if(os.path.isfile(aux_files)):
            nd = np.load(aux_files)
            self.w_in = nd["w_in"]
            self.w_out = nd["w_out"]
            if self.w_in.shape != (self.input_size(), 2):
                self.w_in = np.random.normal(0,1,size=[self.input_size(),2])
            if self.w_out.shape != (2, self.output_size()):
                self.w_out = np.random.normal(0,1,size=[2, self.output_size()])
        else:
            self.w_in = np.random.normal(0,1,size=[self.input_size(),2])
            self.w_out = np.random.normal(0,1,size=[2, self.output_size()])


    def store_tw(self,filename):
        aux_files = filename.replace(".bnn",".npz")

        self.lif.WriteToFile(filename)
        np.savez(aux_files,w_in=self.w_in,w_out=self.w_out)

    def optimize(self,ts=datetime.timedelta(seconds=60),max_steps=1000000):
        # Break symmetry by adding noise
        self.lif.AddNoise(0.5,15)
        self.lif.AddNoiseVleak(8,8)
        self.lif.AddNoiseGleak(0.2,8)
        self.lif.AddNoiseSigma(0.2,10)
        self.lif.AddNoiseCm(0.1,10)
        self.lif.CommitNoise()

        r_values = np.zeros(1000000)
        r_counter=0

        (current_return,mean_ret) =  self.run_multiple_episodes()
        r_values[r_counter]=mean_ret
        r_counter+=1

        num_distortions = 4
        num_distortions_sigma=3
        num_distortions_vleak=2
        num_distortions_gleak=2
        num_distortions_cm=2
        steps_since_last_improvement=0

        starttime = datetime.datetime.now()
        endtime = starttime + ts
        steps=-1
        log_freq=50 # Changed from 250 to 50 to log more frequently, especially in the early stages of optimization
        while endtime>datetime.datetime.now() and steps < max_steps:
            steps+=1

            self.backup_w_in = self.w_in
            self.backup_w_out = self.w_out

            self.w_in = self.w_in + np.random.normal(size=self.w_in.shape)
            self.w_out = self.w_out + np.random.normal(size=self.w_out.shape)

            # weight
            distortions = rng.randint(0,num_distortions)
            variance = rng.uniform(0.01,0.4)

            # sigma
            distortions_sigma = rng.randint(0,num_distortions_sigma)
            variance_sigma = rng.uniform(0.01,0.05)

            # vleak
            distortions_vleak = rng.randint(0,num_distortions_vleak)
            variance_vleak = rng.uniform(0.1,3)

            # vleak
            distortions_gleak = rng.randint(0,num_distortions_gleak)
            variance_gleak = rng.uniform(0.05,0.5)

            #cm
            distortions_cm = rng.randint(0,num_distortions_cm)
            variance_cm = rng.uniform(0.01,0.1)

            self.lif.AddNoise(variance,distortions)
            self.lif.AddNoiseSigma(variance_sigma,distortions_sigma)
            self.lif.AddNoiseVleak(variance_vleak,distortions_vleak)
            self.lif.AddNoiseCm(variance_cm,distortions_cm)
            self.lif.AddNoiseGleak(variance_gleak,distortions_gleak)

            (new_return,mean_ret) =  self.run_multiple_episodes()
            r_values[r_counter]=mean_ret
            r_counter+=1
            # print('Stochastic Return: '+str(new_return))
            if(new_return > current_return):
                # print('Improvement! New Return: '+str(new_return))
                if(self.logfile != None):
                    elapsed = datetime.datetime.now()-starttime
                    self.logfile.write('Improvement after: '+str(steps)+' steps, with return '+str(new_return)+', Elapsed: '+str(elapsed.total_seconds())+'\n')
                    self.logfile.flush()

                current_return=new_return
                self.lif.CommitNoise()
                steps_since_last_improvement=0

                num_distortions-=1
                if(num_distortions<4):
                    num_distortions=4

                num_distortions_sigma-=1
                if(num_distortions_sigma<3):
                    num_distortions_sigma=3

                num_distortions_vleak-=1
                if(num_distortions_vleak<2):
                    num_distortions_vleak=2

                num_distortions_gleak-=1
                if(num_distortions_gleak<2):
                    num_distortions_gleak=2

                num_distortions_cm-=1
                if(num_distortions_cm<2):
                    num_distortions_cm=2
                # print('Set Distortion to '+str(num_distortions))
            else:
                steps_since_last_improvement+=1
                self.lif.UndoNoise()

                self.w_in = self.backup_w_in
                self.w_out = self.backup_w_out

                # no improvement seen for 100 steps
                if(steps_since_last_improvement>50):
                    steps_since_last_improvement=0

                    # reevaluate return
                    (current_return,mean_ret) =  self.run_multiple_episodes()
                    r_values[r_counter]=mean_ret
                    r_counter+=1
                    # print('Reevaluate to: '+str(current_return))
                    if(self.logfile != None):
                        self.logfile.write('Reevaluate after: '+str(steps)+' steps, with return '+str(new_return)+'\n')
                        self.logfile.flush()


                    # Increase variance
                    num_distortions+=1
                    if(num_distortions>12):
                        num_distortions=12
                    # Increase variance sigma
                    num_distortions_sigma+=1
                    if(num_distortions_sigma>8):
                        num_distortions_sigma=8
                    # Increase variance vleak
                    num_distortions_vleak+=1
                    if(num_distortions_vleak>6):
                        num_distortions_vleak=6
                    # Increase variance vleak
                    num_distortions_gleak+=1
                    if(num_distortions_gleak>6):
                        num_distortions_gleak=6
                    # Increase variance cm
                    num_distortions_cm+=1
                    if(num_distortions_cm>4):
                        num_distortions_cm=4
            if(steps % log_freq == 0 and self.csvlogfile != None):
                elapsed = datetime.datetime.now()-starttime
                avg_cost = self.evaluate_avg()
                performance_r = np.mean(r_values[0:r_counter])
                self.csvlogfile.write(str(steps)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
                self.csvlogfile.flush()
                # outfile = logdir+'/tw-'+str(worker_id)+'_steps-'+str(steps)+'.bnn'
                # lif.WriteToFile(outfile)
                    # print('Set Distortion to '+str(num_distortions))
        if(self.logfile != None):
            self.logfile.write('Total steps done: '+str(steps)+'\n')
            self.logfile.close()
        if(self.csvlogfile != None):
            elapsed = datetime.datetime.now()-starttime
            avg_cost = self.evaluate_avg()
            performance_r = np.mean(r_values[0:r_counter])
            self.csvlogfile.write(str(steps)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
            self.csvlogfile.flush()

    def replay(self,filename):
        self.load_tw(filename)
        if not os.path.exists('vid'):
            os.makedirs('vid')
        print('Average Reward: '+str(self.evaluate_avg()))
        print('Replay Return: '+str(self.run_multiple_episodes()))
        try:
            self.run_one_episode(True)
        finally:
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None


    def replay_arg(self):

        worker_id =1
        if(len(sys.argv)>1):
            worker_id = int(sys.argv[1])

        filename = 'bnn1/tw-optimized_'+str(worker_id)+'.bnn'
        self.load_tw(filename)

        print('Replay Return: '+str(self.run_multiple_episodes()))

        self.run_one_episode(True)

    def optimize_and_store(self,worker_id,in_file='tw_pure.bnn'):
        self.load_tw(in_file)
        
        if(worker_id.isdigit()):
            seed = int(worker_id)+20*datetime.datetime.now().microsecond+23115
        else:
            seed = 20*datetime.datetime.now().microsecond+23115

        self.lif.SeedRandomNumberGenerator(seed)
        rng.seed(seed)

        root_path = 'results/filter_'+str(self.filter_len)+'_'+str(self.mean_len)
        log_path = root_path+'/logs_csv'
        log_path_txt = root_path+'/logs_txt'
        store_path = root_path+'/final'

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(log_path_txt):
            os.makedirs(log_path_txt)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        log_file=log_path_txt+'/textlog_'+worker_id+'.log'
        csv_log=log_path+'/csvlog_'+worker_id+'.log'
        self.logfile = open(log_file, 'w')
        self.csvlogfile = open(csv_log, 'w')


        print('Begin Return of '+worker_id+': '+str(self.run_multiple_episodes()))
        self.optimize(ts=datetime.timedelta(hours=6),max_steps=20000)
        print('End Return: of '+worker_id+': '+str(self.run_multiple_episodes()))

        outfile = store_path+'/tw-optimized_'+worker_id+ '.bnn'

        self.store_tw(outfile)
    
    def optimize_and_store_experiment(self, seed, in_file='tw_pure.bnn', experiment_name='experiment'):
        self.load_tw(in_file)
        
        self.lif.SeedRandomNumberGenerator(seed)
        rng.seed(seed)
        np.random.seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

        root_path = os.path.join(CURRENT_DIR, 'results', experiment_name)
        os.makedirs(root_path, exist_ok=True)

        log_file = root_path + '/textlog_' + str(seed) + '.log'
        csv_log = root_path + '/csvlog_' + str(seed) + '.log'
        self.logfile = open(log_file, 'w')
        self.csvlogfile = open(csv_log, 'w')

        try:
            print('Begin Return of ' + str(seed) + ': ' + str(self.run_multiple_episodes()))
            self.optimize(ts=datetime.timedelta(hours=6), max_steps=20000)
            print('End Return of ' + str(seed) + ': ' + str(self.run_multiple_episodes()))

            outfile = root_path + '/tw-optimized_' + str(seed) + '.bnn'
            self.store_tw(outfile)
        finally:
            if self.logfile is not None:
                self.logfile.close()
                self.logfile = None
            if self.csvlogfile is not None:
                self.csvlogfile.close()
                self.csvlogfile = None



def experiment(partial_obs=False):
    seeds = [0,1,2,3,4]
    filter_len = 10
    mean_len = 5
    experiment_name = 'experiment_partial' if partial_obs else 'experiment_full'
    root_path = os.path.join(CURRENT_DIR, 'results', experiment_name)

    os.makedirs(root_path, exist_ok=True)

    processes = []
    for seed in seeds:
        print('Starting run with seed: ' + str(seed))
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            '--experiment-seed', str(seed),
            '--filter', str(filter_len),
            '--mean', str(mean_len),
        ]
        if partial_obs:
            cmd.append('--partial-obs')
        processes.append((seed, subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))))

    for seed, process in processes:
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError('Experiment run failed for seed {} with exit code {}'.format(seed, return_code))

    curves = []
    steps = None

    for seed in seeds:
        csv_path = root_path + '/csvlog_' + str(seed) + '.log'
        current_steps = []
        eval_returns = []
        with open(csv_path, 'r') as handle:
            for line in handle:
                parts = line.strip().split(';')
                if len(parts) < 2:
                    continue
                current_steps.append(int(parts[0]))
                eval_returns.append(float(parts[1]))
        if steps is None:
            steps = current_steps
        elif steps != current_steps:
            raise ValueError('CSV logs do not share the same checkpoints.')
        curves.append(eval_returns)

    curve_array = np.array(curves)
    mean_values = np.mean(curve_array, axis=0)
    std_values = np.std(curve_array, axis=0)
    output_path = root_path + '/experiment_mean_return.png'
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_values, label='Average over {} seeds'.format(len(seeds)))
    plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.25)
    plt.xlabel('Search iterations')
    plt.ylabel('Mean return over 50 eval episodes')
    if partial_obs:
        plt.title('HalfCheetah with Partial Obs. - ONC with ARS')
    else:
        plt.title('HalfCheetah - ONC with ARS')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    final_returns = curve_array[:, -1]
    print('Final eval return: {:.1f} +- {:.1f}'.format(np.mean(final_returns), np.std(final_returns)))
    print('Saved experiment plot to ' + output_path)

    summary_path = root_path + '/final_return_summary.txt'
    with open(summary_path, 'w') as handle:
        handle.write('Experiment: ' + experiment_name + '\n')
        handle.write('Metric: final avg_cost from each csvlog (mean return over 50 eval episodes)\n\n')
        for seed, value in zip(seeds, final_returns):
            handle.write('Seed {}: {}\n'.format(seed, value))
        handle.write('\n')
        handle.write('Final mean return: {:.6f}\n'.format(np.mean(final_returns)))
        handle.write('Final std return: {:.6f}\n'.format(np.std(final_returns)))
        handle.write('Final mean +- std: {:.1f} +- {:.1f}\n'.format(np.mean(final_returns), np.std(final_returns)))


def demo_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter',default=10,type=int)
    parser.add_argument('--mean',default=5,type=int)
    parser.add_argument('--file',default="tw_pure.bnn")
    parser.add_argument('--optimize',action="store_true")
    parser.add_argument('--id',default="0")
    parser.add_argument('--record-video')
    args = parser.parse_args()

    render_mode = None if args.optimize else "human"
    if args.record_video:
        render_mode = "rgb_array"
    env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    # print('Observation space: '+str(env.observation_space.shape[0]))
    # print('Action space: '+str(env.action_space.shape[0]))

    twenv = TWsearchEnv(env,args.filter,args.mean,args.record_video)
    if(args.optimize):
        print("Optimize")
        twenv.optimize_and_store(str(args.id),args.file)
    else:
        print("Replay")
        twenv.replay(args.file)


if __name__ == "__main__":
    if '--experiment-seed' in sys.argv:
        seed_index = sys.argv.index('--experiment-seed')
        seed = int(sys.argv[seed_index + 1])
        partial_obs = '--partial-obs' in sys.argv
        filter_len = 10
        mean_len = 5
        if '--filter' in sys.argv:
            filter_index = sys.argv.index('--filter')
            filter_len = int(sys.argv[filter_index + 1])
        if '--mean' in sys.argv:
            mean_index = sys.argv.index('--mean')
            mean_len = int(sys.argv[mean_index + 1])

        env = gym.make("HalfCheetah-v5", render_mode=None)
        twenv = TWsearchEnv(env, filter_len, mean_len, partial_obs=partial_obs)
        try:
            experiment_name = 'experiment_partial' if partial_obs else 'experiment_full'
            twenv.optimize_and_store_experiment(seed, experiment_name=experiment_name)
        finally:
            env.close()
    elif '--experiment_full' in sys.argv:
        experiment()
    elif '--experiment_partial' in sys.argv:
        experiment(partial_obs=True)
    else:
        demo_run()
