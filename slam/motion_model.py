from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

class MotionModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def probability(self,new_pose, prev_pose, u):
        pass

    @abstractmethod
    def sample(self, prev_pose, u):
        pass


class OdometryModel(MotionModel):

    def __init__(self, **kwargs):
        self.alpha1 = kwargs.get("alpha1", 0.1)
        self.alpha2 = kwargs.get("alpha2", 0.1)
        self.alpha3 = kwargs.get("alpha3", 0.1)
        self.alpha4 = kwargs.get("alpha4", 0.1)
        self.eps = 1e-3

    def probability(self,new_pose, prev_pose, u):
        od_prev = u[0,:]
        od_new = u[1,:]

        delta_rot1 = np.arctan2(od_new[1]-od_prev[1],od_new[0]-od_new[0]) - od_prev[2]
        delta_trans = np.linalg.norm(od_prev[0:1]-od_new[0:1])
        delta_rot2 = od_new[2] - od_prev[2] - delta_rot1

        delta_rot1_pose = np.arctan2(new_pose[1] - prev_pose[1], new_pose[0] - prev_pose[0]) - prev_pose[2]
        delta_trans_pose = np.linalg.norm(prev_pose[0:1] - new_pose[0:1])
        delta_rot2_pose = new_pose[2] - prev_pose[2] - delta_rot1_pose

        std1 = self.alpha1*delta_rot1_pose**2+ self.alpha2*delta_trans_pose**2
        err1 = delta_rot1-delta_rot1_pose
        p1 = norm.cdf(err1+self.eps,scale=std1) - norm.cdf(err1-self.eps,scale=std1)

        std2 = self.alpha3*delta_trans_pose**2+self.alpha4*(delta_rot1_pose**2 + delta_rot2_pose**2)
        err2 = delta_trans-delta_trans_pose
        p2 = norm.cdf(err2+self.eps,scale=std2) - norm.cdf(err2-self.eps,scale=std2)

        err3 = delta_rot2-delta_rot2_pose
        p3 = norm.cdf(err3+self.eps,scale=std1)-norm.cdf(err3-self.eps,scale=std1)

        return p1*p2*p3

    def sample(self, prev_pose, u):
        pose = np.zeros(3)

        od_prev = u[0, :]
        od_new = u[1, :]

        delta_rot1 = np.arctan2(od_new[1] - od_prev[1], od_new[0] - od_new[0]) - od_prev[2]
        delta_trans = np.linalg.norm(od_prev[0:1] - od_new[0:1])
        delta_rot2 = od_new[2] - od_prev[2] - delta_rot1

        delta_rot1_hat = delta_rot1 - norm.rvs(scale=(self.alpha1*delta_rot1**2 + self.alpha2*delta_trans**2))
        delta_trans_hat = delta_trans - norm.rvs(scale=(self.alpha3*delta_trans**2 + self.alpha4*(delta_rot1**2 + delta_rot2**2)))
        delta_rot2_hat = delta_rot2 - norm.rvs(scale=(self.alpha1*delta_rot1**2 +self.alpha2*delta_trans**2))

        pose[0] = prev_pose[0] + delta_trans_hat*np.cos(prev_pose[2]+delta_rot1_hat)
        pose[1] = prev_pose[1] + delta_trans_hat*np.sin(prev_pose[2]+delta_rot1_hat)
        pose[2] = prev_pose[2] + delta_rot2_hat + delta_rot1_hat

        return pose




def test():
    model = OdometryModel()

    prev_pose = np.array([1,1,0])
    u = np.array([[0,0,0],[0.5,0,0]])
    new_pose = np.array([5,0,0])

    print(model.probability(new_pose,prev_pose,u))
    print(model.sample(prev_pose,u))

if __name__ == "__main__":
    test()
