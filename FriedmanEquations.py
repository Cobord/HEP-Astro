import numpy as np
import tensorflow as tf

tfG = tf.constant(6.67408 * 10**(-11))
tfc = tf.constant(3.0 * 10**8)
LambdaEst = 0.0
Omega0REst = 0.0
Omega0MEst = 0.0
Omega0kEst = 0.0
Omega0LambdaEst = 0.0
Lambda = tf.get_variable("Lambda",dtype=tf.float32,initializer=tf.constant(LambdaEst))
Omega0R = tf.get_variable("Omega0R",dtype=tf.float32,initializer=tf.constant(Omega0REst))
Omega0M = tf.get_variable("Omega0M",dtype=tf.float32,initializer=tf.constant(Omega0MEst))
Omega0k = tf.get_variable("Omega0k",dtype=tf.float32,initializer=tf.constant(Omega0kEst))
Omega0Lambda = tf.get_variable("Omega0Lambda",dtype=tf.float32,initializer=tf.constant(Omega0LambdaEst))

kEst = 0.0
matterEst = .3
radEst = .1
lambdaEst = .2
tfMatter = tf.get_variable("A",dtype=tf.float32,initializer=tf.constant(matterEst))
tfRad = tf.get_variable("B",dtype=tf.float32,initializer=tf.constant(radEst))
tfLambda = tf.get_variable("C",dtype=tf.float32,initializer=tf.constant(lambdaEst))
tfK = tf.get_variable("k",dtype=tf.float32,initializer=tf.constant(kEst))

pastTimeInterval = np.linspace(0,4.415*10**(17),num=5,endpoint=False)
futureTimeInterval = np.linspace(0,2*10**(17),num=5,endpoint=False)
fullTimeInterval=np.flip(pastTimeInterval,0)*(-1)
fullTimeInterval=np.hstack([fullTimeInterval,futureTimeInterval])

def myDiffEq(a,past=True):
    rho = tfMatter*tf.pow(a,tf.constant(-3.0))+tfRad*tf.pow(a,tf.constant(-4.0))+tfLambda
    RHS=tf.sqrt((tf.constant(8*np.pi/3)*tfG*rho-tfK*tf.pow(tfc,2)*tf.pow(a,-2.0))*tf.pow(a,2.0))
    if (past):
        return tf.constant(-1.0)*RHS
    else:
        return RHS

pastA=tf.contrib.integrate.odeint(lambda a,_: myDiffEq(a,past=True),1.0,pastTimeInterval)
futureA=tf.contrib.integrate.odeint(lambda a,_: myDiffEq(a,past=False),1.0,futureTimeInterval)
fullA = tf.reverse(pastA,axis=tf.constant(np.array([0])))
fullA = tf.concat([fullA,futureA],0)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(fullA))