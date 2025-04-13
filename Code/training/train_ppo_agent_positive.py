# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import time
import os
import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import *
import re


def _l1_norm(model, lambda_l1):
    l1_loss = 0
    for name, param in model.named_parameters():
        # Only apply L1 regularization to input weights of GRU (weight_ih_l0)
        if 'weight_ih_l0' in name and "bias" not in name:
            l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


def train_ppo_positive(envs: gym.vector.SyncVectorEnv, args, model_file_name, device, writer=None, logger=None, seed=None):
    hidden_size = args.hidden_size
    l1_lambda = args.l1_lambda
    if not seed:
        seed = args.seed

    feature_extractor = False if "noFE" in args.exp_name else True

    if args.ppo_type == "original":
        from agents import PPOAgent
        agent = PPOAgent(envs, hidden_size=hidden_size, feature_extractor=feature_extractor, arch_details=args.exp_name).to(device)
    elif args.ppo_type == "lstm":
        from agents import LstmAgent
        agent = LstmAgent(envs, h_size=hidden_size).to(device)
    elif args.ppo_type == "gru":
        from agents import GruAgent
        agent = GruAgent(envs, h_size=hidden_size, feature_extractor=feature_extractor).to(device)
    else:
        raise NotImplementedError


    if args.ppo_type == "original":
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:   # LSTM or GRU
        optimizer = optim.Adam([
            {'params': agent.critic.parameters(), 'lr': args.value_learning_rate, 'name':'value'},
            {'params': [p for name, p in agent.named_parameters() if "critic" not in name], 'lr': args.learning_rate, 'eps':1e-5, 'weight_decay':args.weight_decay, 'name':'other'}
        ])

    # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    positive_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.ppo_type == 'lstm':
        next_rnn_state = (
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    elif args.ppo_type == 'gru':
        next_rnn_state = torch.zeros(agent.gru.num_layers, args.num_envs, agent.gru.hidden_size).to(device)


    # for iteration in range(1, args.num_iterations + 1):
    while global_step < args.total_timesteps:
        iteration = args.num_iterations

        obs = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []

        if args.ppo_type == 'gru':
            initial_rnn_state = next_rnn_state.clone()
        elif args.ppo_type == 'lstm':
            initial_rnn_state = (next_rnn_state[0].clone(), next_rnn_state[1].clone())

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            # frac = 1.0 - (iteration - 1.0) / args.num_iterations
            frac = 1.0 - (global_step - 1.0) / args.total_timesteps
            if args.ppo_type == "original":
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            else:   # LSTM or GRU
                lr_value = frac * args.value_learning_rate
                lr_other = frac * args.learning_rate
                for param_group in optimizer.param_groups:
                    if param_group.get('name') == 'value':
                        param_group['lr'] = lr_value
                    elif param_group.get('name') == 'other':
                        param_group['lr'] = lr_other

        positive_example = False
        number_samples = 0

        for step in range(0, args.num_steps):
            # obs[step] = next_obs
            # dones[step] = next_done
            obs.append(next_obs)
            dones.append(next_done)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.ppo_type == 'gru':
                    action, logprob, _, value, next_rnn_state = agent.get_action_and_value(next_obs, next_rnn_state, next_done)
                elif args.ppo_type == 'lstm':
                    action, logprob, _, value, next_rnn_state = agent.get_action_and_value(next_obs, next_rnn_state, next_done)
                else:   # original
                    action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                # values[step] = value.flatten()
                values.append(value.flatten())

            actions.append(action)
            logprobs.append(logprob)
            # actions[step] = action
            # logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # next_done = np.logical_or(terminations, truncations)
            next_done = terminations
            # rewards[step] = torch.tensor(reward).to(device).view(-1)
            rewards.append(torch.tensor(reward).to(device).view(-1))
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # if next_done == 1.0:
            #     positive_example = True
            #     number_samples = len(obs)
            #     break


            # if positive_example:
            #     positive_step += number_samples

            global_step += len(obs)
            number_samples = len(obs)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # if not positive_example:
        #     continue


        rewards = torch.cat(rewards)
        values = torch.cat(values)
        # bootstrap value if not done
        with torch.no_grad():
            if args.ppo_type == 'gru':
                next_value = agent.get_value(next_obs, next_rnn_state, next_done).reshape(1, -1)
            elif args.ppo_type == 'lstm':
                next_value = agent.get_value(next_obs, next_rnn_state, next_done).reshape(1, -1)
            else:   # original
                next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(number_samples)):
                if t == number_samples - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_dones = dones.reshape(-1) # done is used for GAE
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        b_obs = torch.cat(obs).reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = torch.cat(logprobs).reshape(-1)
        b_actions = torch.cat(actions).reshape((-1,) + envs.single_action_space.shape)
        b_dones = torch.cat(dones).reshape(-1) # done is used for GAE
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        # Optimizing the policy and value network
        if args.ppo_type == "original":
            # b_inds = np.arange(args.batch_size)
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(number_samples).reshape(number_samples, args.num_envs)
            clipfracs = []
            for epoch in range(args.update_epochs):
                # np.random.shuffle(b_inds)
                # for start in range(0, args.batch_size, args.minibatch_size):
                #     end = start + args.minibatch_size
                #     mb_inds = b_inds[start:end]
                np.random.shuffle(envinds)  # TODO: this is wrong!! envinds is 1!!
                for start in range(0, args.num_envs, envsperbatch): # if num_envs==1 this loop runs only once
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        if len(mb_advantages) > 1:
                            std = mb_advantages.std()
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                        else:
                            logger.info("Skipping normalization for single-element mb_advantages.")

                    # L1 loss
                    # l1_loss = _l1_norm(model=agent.actor, lambda_l1=args.l1_lambda)
                    l1_loss = agent.get_l1_norm() if feature_extractor else agent.get_l1_norm_actor()
                    # l1_loss = agent.get_l1_norm_actor()

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    # pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # pg_loss = torch.max(pg_loss1, pg_loss2).mean() + l1_loss
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean() + l1_loss * l1_lambda


                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()

                    # TODO: check
                    l1_reg = torch.tensor(0.).to(device)
                    for param in agent.actor.parameters():
                        l1_reg += torch.norm(param, 1)

                    # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + l1_lambda * l1_reg
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef


                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
        
        else:   # LSTM or GRU
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(number_samples).reshape(number_samples, args.num_envs)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    if args.ppo_type == 'gru':
                        # print('mb_inds:', mb_inds, 'b_obs:', b_obs[mb_inds], 'initial_rnn_state:', initial_rnn_state[:, mbenvinds], 'b_dones:', b_dones[mb_inds], 'b_actions:', b_actions.long()[mb_inds])
                        _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                            b_obs[mb_inds],
                            initial_rnn_state[:, mbenvinds],
                            b_dones[mb_inds],
                            b_actions.long()[mb_inds],
                        )
                    elif args.ppo_type == 'lstm':
                        _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                            b_obs[mb_inds],
                            (initial_rnn_state[0][:, mbenvinds], initial_rnn_state[1][:, mbenvinds]),
                            b_dones[mb_inds],
                            b_actions.long()[mb_inds],
                        )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        if len(mb_advantages) > 1:
                            std = mb_advantages.std()
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                        else:
                            logger.info("Skipping normalization for single-element mb_advantages.")

                    #L1 loss
                    if args.ppo_type == 'gru':
                        l1_loss = agent.get_l1_norm()
                    elif args.ppo_type == 'lstm':
                        l1_loss = _l1_norm(model=agent.lstm, lambda_l1=args.l1_lambda)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean() + l1_lambda * l1_loss

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                    entropy_loss = entropy.mean()

                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        if args.ppo_type == "original": writer.add_scalar("losses/l1_reg", l1_reg.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if iteration % 1000 == 0:
            logger_flush(logger)


    print("args:", args)
    print(f"Positive steps: {positive_step}")

    if "sweep" in args.exp_name:
        pattern = r"^(.*?)_SD"
        result = re.match(pattern, args.exp_name)
        sweep_directory = result.group(1) + "/" + model_file_name
        os.makedirs(os.path.dirname(sweep_directory), exist_ok=True)

    envs.close()
    writer.close()
    logger.info(f"Experiment: {args.exp_name}")
    if "test" not in args.exp_name:
        torch.save(agent.state_dict(), sweep_directory) if "sweep" in args.exp_name else torch.save(agent.state_dict(), model_file_name)
        logger.info(f"Saved on {model_file_name}")
    else:
        print("Test mode, not saving model")
