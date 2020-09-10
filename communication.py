
from abc import ABC, abstractmethod
from time import sleep
from threading import Thread
from random import seed, randint

class CommunicationNode(ABC):
    def __init__(self):
        super().__init__()  

    @abstractmethod
    def recv_msg(self, msg):
        pass

class CommunicationChannel():

  def __init__(self, com_filter = lambda sender, recipient: True, **kwargs):
    self.com_filter = com_filter
    self.delay = kwargs["delay"] if "delay" in kwargs else None
    self.packet_loss = kwargs["packet_loss"] if "packet_loss" in kwargs else None
    seed()
  
  def send_msg(self, sender, recipients, msg):
    for recipient in list(filter(lambda recipient: self.com_filter(sender, recipient), recipients)):
      t = Thread(target=self.__send_msg_aux, args=(recipient, msg))
      t.start()
  
  def __send_msg_aux (self, recipient, msg):
      if not self.delay is None:
        sleep(self.delay)
      if self.packet_loss is None or randint(0, 100-1) > self.packet_loss:
        recipient.recv_msg(msg)
      else:
        recipient.recv_msg(None)
