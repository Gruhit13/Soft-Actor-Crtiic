import numpy as np
from typing import Tuple, Union

class ReplayBuffer:
	def __init__(self, obs_shape: Tuple[int], n_action: int, max_exp_size: int):

		self.max_exp_size = max_exp_size
		self.state = np.empty((self.max_exp_size, *obs_shape), dtype=np.float32)
		self.action = np.empty((self.max_exp_size, n_action), dtype=np.float32)
		self.reward = np.empty((self.max_exp_size, 1), dtype=np.float32)
		self.done = np.empty((self.max_exp_size, 1), dtype=bool)
		self.next_state = np.empty((self.max_exp_size, *obs_shape), dtype=np.float32)

		self.cntr = 0
		self.max_buffer = 0

	def store_experience(self, s: np.ndarray, a: Union[np.ndarray, float], r: float, d: bool, n_s: np.ndarray) -> None:
		self.state[self.cntr] = s
		self.action[self.cntr] = a
		self.reward[self.cntr] = r
		self.done[self.cntr] = d
		self.next_state[self.cntr] = n_s

		self.cntr = (self.cntr + 1) % self.max_exp_size

		self.max_buffer = max(self.cntr, self.max_buffer)

	def sample(self, batch_size: int) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

		if batch_size > self.max_buffer:
			return

		indx = np.random.randint(0, self.max_buffer, (batch_size,))

		return (
				self.state[indx],
				self.action[indx],
				self.reward[indx],
				self.done[indx],
				self.next_state[indx]
			)