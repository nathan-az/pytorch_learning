import math
from collections import namedtuple

from PIL import Image
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as T
import torch
import torch.nn.functional as F

resize = T.Compose(
    [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
)

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_num_actions(env):
    return env.action_space.n


def get_cart_location(env, screen_width):
    """
    Returns the position of the middle of the cart
    :param env:
    :param screen_width:
    :return:
    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env, device=get_device()):
    # transpose (h,w,c) to (c,h,w)
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    # trim screen, don't need all height or width since game fails with too much sway vertically and cart in lower half
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(
            cart_location - view_width // 2, cart_location + view_width // 2
        )

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


def select_action(
    state, policy_net, n_actions, steps_done, config, device=get_device()
):
    sample = random.random()
    eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(
        -1.0 * steps_done / config.EPS_DECAY
    )
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)


def optimise_model(
    policy_net, target_net, optimiser, memory, config, device=get_device()
):
    if len(memory) < config.BATCH_SIZE:
        return

    transitions = memory.sample(config.BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).to(device).gather(1, action_batch)

    next_state_values = torch.zeros(config.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states.to(device)).max(1)[0].detach()
    )

    expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    optimiser.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimiser.step()


def train_model(
    num_episodes,
    env,
    policy_net,
    target_net,
    optimiser,
    memory,
    config,
    enable_plot_durations=False,
    episode_durations=[],
    device=get_device(),
):
    steps_done = 0
    n_actions = get_num_actions(env)
    for i in range(num_episodes):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            action = select_action(
                state, policy_net, n_actions, steps_done, config, device=get_device()
            )
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen(env)

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            memory.push(state, action, reward, next_state)

            state = next_state

            optimise_model(policy_net, target_net, optimiser, memory, config)

            steps_done += 1

            if done:
                episode_durations.append(t + 1)
                if enable_plot_durations:
                    plot_durations(episode_durations)
                break

        if i % config.TARGET_UPDPATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # print(
        #     f"[{i}/{num_episodes}]: {np.mean(episode_durations[-1 * max(len(episode_durations), 10):])}"
        # )