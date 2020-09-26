import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt

import gym

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()


TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])

import numpy as np
import math,random
class ActorNet2(object):
  
  def __init__(self, input_size, hidden_size,output_size, std=5e-1):
    
    print("An actor network is created.")
    
    self.params = {}
    self.params['W1'] = self._uniform_init(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = self._uniform_init(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    self.params['W3'] = self._uniform_init(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)
    
    self.optm_cfg ={}
    self.optm_cfg['W1'] = None
    self.optm_cfg['b1'] = None
    self.optm_cfg['W2'] = None
    self.optm_cfg['b2'] = None
    self.optm_cfg['W3'] = None
    self.optm_cfg['b3'] = None
    


  def evaluate_gradient(self, s, a, olp, adv,b, clip_param, max_norm):
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
        
    batch_size, _ = s.shape
  
    z1=np.dot(s,W1)+b1
    H1=np.maximum(0,z1) 
    z2=np.dot(H1,W2)+b2
    mu=b * np.tanh(z2) 

    z3=np.dot(H1,W3)+b3
    sigma=np.log(1+np.exp(z3))
    
    alp=np.zeros((batch_size,len(a[0])))
    ratio=np.zeros((batch_size,len(a[0])))
    surr1=np.zeros((batch_size,len(a[0])))
    surr2=np.zeros((batch_size,len(a[0])))
    mu_derv=np.zeros((batch_size,len(a[0])))
    sigma_derv=np.zeros((batch_size,len(a[0])))
    
    for i in range(batch_size):
        for j in range(len(a[0])):
            alp[i,j]=-((a[i,j] - mu[i,j]) ** 2) / (2 * sigma[i,j]**2) - np.log(sigma[i,j]) - np.log(np.sqrt(2 * np.pi))
            ratio[i,j]= np.exp(alp[i,j]-olp[i,j])
    
            surr1[i,j]=ratio[i,j]*adv[i]
            surr2[i,j]= np.clip(ratio[i,j],1-clip_param,1+clip_param)*adv[i]
                    
            if(surr2[i,j]<surr1[i,j] and (ratio[i,j]<1-clip_param or ratio[i,j]>1+clip_param)):
                mu_derv[i,j]=0
                sigma_derv[i,j]=0
            else:
                mu_derv[i,j]=-(b*adv[i]*math.exp(-(a[i,j]-b*math.tanh(z2[i,j]))**2/(2*sigma[i,j]**2)-olp[i,j])*(1-(math.tanh(z2[i,j]))**2)*(a[i,j]-b*math.tanh(z2[i,j])))/(math.sqrt(2*math.pi)*sigma[i,j]**3)
                sigma_derv[i,j]=(adv[i]*math.exp(-(a[i,j]-mu[i,j])**2/(2*math.log(math.exp(z3[i,j])+1)**2)+z3[i,j]-olp[i,j])*(math.log(math.exp(z3[i,j])+1)**2-mu[i,j]**2+2*a[i,j]*mu[i,j]-a[i,j]**2))/(math.sqrt(2*math.pi)*(math.exp(z3[i,j])+1)*math.log(math.exp(z3[i,j])+1)**4)
        
            

    grads = {}

    out1=sigma_derv.dot(W3.T)
    out1+=mu_derv.dot(W2.T)
    
    out1[z1<=0]=0

    sigma_derv/=len(a[0])
    mu_derv/=len(a[0])
    grads['W3']=np.dot(H1.T, sigma_derv)/batch_size
    grads['W2']=np.dot(H1.T, mu_derv)/batch_size
    grads['W1']=np.dot(s.T, out1)/batch_size
    grads['b3']=np.sum(sigma_derv, axis=0)/batch_size
    grads['b2']=np.sum(mu_derv, axis=0)/batch_size
    grads['b1']=np.sum(out1, axis=0)/batch_size
    
    
    total_norm = np.sqrt((grads['W3']**2).sum()+(grads['W2']**2).sum()+(grads['W1']**2).sum()
    +(grads['b3']**2).sum()+(grads['b2']**2).sum()+(grads['b1']**2).sum())
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads['W3']*=clip_coef
        grads['W2']*=clip_coef
        grads['W1']*=clip_coef
        grads['b3']*=clip_coef
        grads['b2']*=clip_coef
        grads['b1']*=clip_coef
    return  grads

  def train(self, s, a, olp, adv,b, clip_param, max_grad_norm):
     # Compute out and gradients using the current minibatch
    grads = self.evaluate_gradient(s, a, olp, adv,b, clip_param, max_grad_norm)
    # Update the weights using adam optimizer
    
    self.params['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[0]
    self.params['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[0]
    self.params['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[0]
    self.params['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[0]
    self.params['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[0]
    self.params['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[1]
    self.optm_cfg['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[1]
    self.optm_cfg['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[1]
    self.optm_cfg['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[1]
    self.optm_cfg['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[1]
    self.optm_cfg['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[1]

    
  def predict(self, s, b):

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    z1=np.dot(s,W1)+b1
    H1=np.maximum(0,z1) 
    z2=np.dot(H1,W2)+b2
    mu=b * np.tanh(z2) 

    z3=np.dot(H1,W3)+b3
    sigma=np.log(1+np.exp(z3))
    
    mu=np.array([mu])
    sigma=np.array([sigma])
    
    a=np.zeros((1,2))
    alp=np.zeros((1,2))
    for j in range(2):
        a[0,j] = mu[0,j] + sigma[0,j] * math.sqrt(-2.0 * math.log(random.random())) *math.sin(2.0 * math.pi * random.random())
        alp[0,j] =-((a[0,j] - mu[0,j]) ** 2) / (2 * sigma[0,j] ** 2) - np.log(sigma[0,j]) - math.log(math.sqrt(2 * math.pi))
        
    return a,alp

  def _adam(self, x, dx, config=None):
      if config is None: config = {}
      config.setdefault('learning_rate', 1e-4)
      config.setdefault('beta1', 0.9)
      config.setdefault('beta2', 0.999)
      config.setdefault('epsilon', 1e-8)
      config.setdefault('m', np.zeros_like(x))
      config.setdefault('v', np.zeros_like(x))
      config.setdefault('t', 0)
      
      next_x = None
      
      #Adam update formula,                                                 #
      config['t'] += 1
      config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
      config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)
      mb = config['m'] / (1 - config['beta1']**config['t'])
      vb = config['v'] / (1 - config['beta2']**config['t'])
    
      next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
      return next_x, config
  
  def _uniform_init(self, input_size, output_size):
      u = np.sqrt(1./(input_size*output_size))
      return np.random.uniform(-u, u, (input_size, output_size))


class CriticNet2(object):
  
  def __init__(self, input_size, hidden_size,output_size, std=5e-1):
    
    print("An actor network is created.")
    
    self.params = {}
    self.params['W1'] = self._uniform_init(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = self._uniform_init(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    
    self.optm_cfg ={}
    self.optm_cfg['W1'] = None
    self.optm_cfg['b1'] = None
    self.optm_cfg['W2'] = None
    self.optm_cfg['b2'] = None
    


  def evaluate_gradient(self, s, target_v, max_norm):
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    
    batch_size, _ = s.shape
  
    z1=np.dot(s,W1)+b1
    H1=np.maximum(0,z1) 
    v=np.dot(H1,W2)+b2
    
    d=np.zeros((batch_size,1))

    for i in range(batch_size):
        d[i,0]=v[i,0]-target_v[i]
        if(d[i,0]<-1):
            d[i,0]=-1
        elif(d[i,0]>1):
            d[i,0]=1
            

    grads = {}

    out1=d.dot(W2.T)

    out1[z1<=0]=0

    grads['W2']=np.dot(H1.T, d)/batch_size
    grads['W1']=np.dot(s.T, out1)/batch_size
    grads['b2']=np.sum(d, axis=0)/batch_size
    grads['b1']=np.sum(out1, axis=0)/batch_size
    
    total_norm = np.sqrt((grads['W2']**2).sum()+(grads['W1']**2).sum()
    +(grads['b2']**2).sum()+(grads['b1']**2).sum())
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads['W2']*=clip_coef
        grads['W1']*=clip_coef
        grads['b2']*=clip_coef
        grads['b1']*=clip_coef
    return  grads

  def train(self, s, target_v, max_grad_norm):
     # Compute out and gradients using the current minibatch
    grads = self.evaluate_gradient(s, target_v, max_grad_norm)
    # Update the weights using adam optimizer
    
    self.params['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[0]
    self.params['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[0]
    self.params['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[0]
    self.params['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[1]
    self.optm_cfg['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[1]
    self.optm_cfg['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[1]
    self.optm_cfg['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[1]

    
  def predict(self, s):

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    
    z1=np.dot(s,W1)+b1
    H1=np.maximum(0,z1) 
    v=np.dot(H1,W2)+b2

    return v

  def _adam(self, x, dx, config=None):
      if config is None: config = {}
      config.setdefault('learning_rate', 3e-4)
      config.setdefault('beta1', 0.9)
      config.setdefault('beta2', 0.999)
      config.setdefault('epsilon', 1e-8)
      config.setdefault('m', np.zeros_like(x))
      config.setdefault('v', np.zeros_like(x))
      config.setdefault('t', 0)
      
      next_x = None
      
      #Adam update formula,                                                 #
      config['t'] += 1
      config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
      config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)
      mb = config['m'] / (1 - config['beta1']**config['t'])
      vb = config['v'] / (1 - config['beta2']**config['t'])
    
      next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
      return next_x, config
  
  def _uniform_init(self, input_size, output_size):
      u = np.sqrt(1./(input_size*output_size))
      return np.random.uniform(-u, u, (input_size, output_size))

class Agent():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 32

    def __init__(self):
        self.training_step = 0
        self.manet = ActorNet2(3,100,2)
        self.mcnet = CriticNet2(3,100,1)
        self.buffer = []
        self.counter = 0   

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        bs = np.array([t.s for t in self.buffer])
        d = [t.a for t in self.buffer]
        ba = np.array([[a[0,0],a[0,1]] for a in d])
        br = np.array([t.r for t in self.buffer])
        bs1 = np.array([t.s_ for t in self.buffer])
        
        d = [t.a_log_p for t in self.buffer]
        bolp = np.array([[a[0,0],a[0,1]] for a in d])

        n=len(bs)
        k=self.batch_size
            
        mean=0
        for i in range(n):
            mean+=br[i]
        mean/=n
        sum=0
        for i in range(n):
            sum+=(br[i]-mean)**2
        std=math.sqrt(sum/(n-1))
        
        target_v = np.zeros((n))
        adv = np.zeros((n))
        
        for i in range(n):
            br[i]=(br[i]-mean)/(std+1e-5)
            
            target_v[i]=br[i] + args.gamma * self.mcnet.predict(bs1)[i,0]
            adv[i] = target_v[i] - self.mcnet.predict(bs)[i,0]
        
        for _ in range(self.ppo_epoch*32):
            
            pool=np.zeros((n))
            result=np.zeros((k)).astype(int)
            for i in range(n):
                pool[i]=i
            for i in range(k):
                j = random.randint(0,n - i-1)
                result[i] = pool[j]
                pool[j] = pool[n - i - 1]
            
            s=np.zeros((k,len(bs[0])))
            a=np.zeros((k,len(ba[0])))
            s1=np.zeros((k,len(bs[0])))
            olp=np.zeros((k,len(ba[0])))
            target_v0=np.zeros((k))
            adv0=np.zeros((k))
            for i in range(k):
                target_v0[i]=target_v[result[i]] 
                adv0[i]=adv[result[i]] 
                for j in range(len(bs[0])):
                    s[i,j]=bs[result[i],j] 
                for j in range(len(ba[0])):
                    a[i,j]=ba[result[i],j] 
                    olp[i,j]=bolp[result[i],j]
                
            self.manet.train(s, a, olp, adv0, 2.0, self.clip_param, self.max_grad_norm)
            self.mcnet.train(s,target_v0, self.max_grad_norm)
            
        del self.buffer[:]


def main():
    env = gym.make('Pendulum-v0')
    env.seed(args.seed)

    agent = Agent()

    training_records = []
    running_reward = -1000
    state = env.reset()
    for i_ep in range(1):
        score = 0
        state = env.reset()

        for t in range(2000000000):
            action, action_log_prob = agent.manet.predict(state,2.0)
            state_, reward, done, _ = env.step([action[0,0].item()])
            if args.render:
                env.render()
            if agent.store(Transition(state, action, action_log_prob, (reward + 8) / 8, state_)):
                agent.update()
            score += reward
            state = state_

            if(t%200==0):
                running_reward = running_reward * 0.9 + score * 0.1
                training_records.append(TrainingRecord(i_ep, running_reward))
            if(t%1000==0):
                print('Ep {}\tMoving average score: {:.2f}\t'.format((int)(t/1000), running_reward))
                score=0

            if running_reward > -200:
                print("Solved! Moving average score is now {}!".format(running_reward))
                env.close()
                
                break

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("ppo.png")
    plt.show()


if __name__ == '__main__':
    main()
