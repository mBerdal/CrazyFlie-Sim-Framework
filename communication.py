from random import seed, randint
from typing import List

class CommunicationNode():
    def __init__(self):
      self.packet_queue = []

    def recv_msg(self, packet):
      self.packet_queue.append(packet)

    def get_msgs(self, step_length):
      packets, r_idxs = [], []
      for i, p in enumerate(self.packet_queue):
        p.delay -= step_length
        if p.delay <= 0:
          packets.append(p)
          r_idxs.append(i)
      for r_idx in r_idxs:
        self.packet_queue.pop(r_idx)
      return packets

class Packet():
  def __init__(self, data_type, data, delay = 0):
    self.data_type = data_type
    self.data = data
    self.delay = delay
  
  def __str__(self):
    return f"data_type: {self.data_type}\ndata: {self.data}\ndelay: {self.delay}"

class CommunicationChannel():


  def __init__(self, com_filter = lambda sender, recipient: True, **kwargs):
    self.com_filter = com_filter
    self.delay = kwargs["delay"] if "delay" in kwargs else 0
    print(f"delay: {self.delay}")
    self.packet_loss = kwargs["packet_loss"] if "packet_loss" in kwargs else None
    seed()
  
  def send_message(self, sender, recipients, msg_type, msg_data):
    for recipient in list(filter(lambda recipient: self.com_filter(sender, recipient), recipients)):
      if self.packet_loss is None or randint(0, 100-1) > self.packet_loss:
        recipient.recv_msg(Packet(msg_type, msg_data, self.delay))