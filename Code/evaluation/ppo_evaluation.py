import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import argparse
import torch
from agents import PPOAgent
import gymnasium as gym
from environment.combogrid_gym import make_env


def run_agent(agent, envs, problem, device):
    print("######### Problem: ", problem, " #########")
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)

    done = False
    total_reward = 0
    num_decisions = 0

    while not done:
        # print("observation: ", obs)
        with torch.no_grad():
            action, _, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
        obs, reward, terminated, truncated, info = envs.step([int(action.cpu().numpy())])
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        total_reward += reward
        done = terminated or truncated
        num_decisions += 1

    print(f'Problem {problem}: Total Reward = {total_reward} in {num_decisions} decisions')
    if "final_info" in info:
        for inf in info["final_info"]:
            if inf and "episode" in inf:
                print("length: ", inf['episode']['l'], "reward: ", inf['episode']['r'])
    

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', default="ppo", type=str)
    parser.add_argument('--game_width', default=3, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--l1', default=0.0, type=float)
    parser.add_argument('--problem', default="All", type=str, help="All, TL-BR, TR-BL, BR-TL, BL-TR")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--log_path', default="logs/", type=str)
    parser.add_argument('--eval_method', nargs='+', default=[None], type=str, help="Nothing but printing trajectories for now")
    parser.add_argument('--total_timesteps', default=50000, type=int)
    parser.add_argument('--ent_coef', default=0.0, type=float)
    parser.add_argument('--clip_coef', default=0.01, type=float)
    parser.add_argument('--cuda', default=True, type=bool)

    parser.add_argument('--options_enabled', default=0, type=int)
    parser.add_argument('--base_model_option', default="nn", type=str)
    parser.add_argument('--num_epoch', default=5000, type=int)
    parser.add_argument('--len3', default=False, type=bool)
    parser.add_argument('--l1_option', default=0.0005, type=float)
    parser.add_argument('--lr_option', default=0.1, type=float)

    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
    envs = []

    if "All" in args.problem:
        for prob in problems:

            model_file_name = f'binary/{args.base_model}-{prob}-gw{args.game_width}-h{args.hidden_size}-l1l{int(args.l1)}-lr{args.lr}-totaltimestep{args.total_timesteps}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_MODEL.pt'
            if args.options_enabled:

                save_path = 'binary/' + args.base_model_option + '_options_list_relu_' + str(args.hidden_size) + '_game_width_' + str(args.game_width) + '_num_epochs_' + str(args.num_epoch) + '_l1_' + str(args.l1_option) + '_lr_' + str(args.lr_option) + '.pkl'
                if args.len3:   save_path = save_path.replace(".pkl", "_onlyws3.pkl")
                with open(save_path, 'rb') as f:
                    options_list = pickle.load(f)
                print(f'\n\n Options list loaded from {save_path} \n')
                options_list = [option for option in options_list if option.problem != prob]


                model_file_name = model_file_name.replace("_MODEL.pt", "_options_MODEL.pt")
                envs = gym.vector.SyncVectorEnv(
                    [make_env(rows=args.game_width, columns=args.game_width, problem=prob, options=options_list) for _ in range(1)],
                ) 
            else:   # TODO: debug this part
                envs = gym.vector.SyncVectorEnv(
                    [make_env(rows=args.game_width, columns=args.game_width, problem=prob) for _ in range(1)],
                )    
            assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

            agent = PPOAgent(envs, hidden_size=args.hidden_size).to(device)
            agent.load_state_dict(torch.load(model_file_name))
            agent.eval()    # running for inference
            run_agent(agent, envs, prob, device)




if __name__ == "__main__":
    main()