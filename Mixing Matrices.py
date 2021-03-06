# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:22:55 2018

@author: Owner
"""

import numpy as np
import tensorflow as tf

doing_CKM=True

if doing_CKM:
    # CKM initialization
    theta12Est = 13.04 * np.pi/180.0
    theta23Est = .201* np.pi/180.0
    theta13Est = 2.38* np.pi/180.0
    deltaCPEst = 1.20
else:
    # PMNS initialization
    theta12Est = 33.62 * np.pi/180.0
    theta23Est = 47.2* np.pi/180.0
    theta13Est = 8.54* np.pi/180.0
    deltaCPEst = 234* np.pi/180.0

theta12 = tf.get_variable("theta_12", dtype=tf.float32, initializer=tf.constant(theta12Est))
theta13 = tf.get_variable("theta_13", dtype=tf.float32, initializer=tf.constant(theta13Est))
theta23 = tf.get_variable("theta_23", dtype=tf.float32, initializer=tf.constant(theta23Est))
deltaCP = tf.get_variable("delta_CP", dtype=tf.float32, initializer=tf.constant(deltaCPEst))
c12 = tf.cos(theta12)
s12 = tf.sin(theta12)
c13 = tf.cos(theta13)
s13 = tf.sin(theta13)
c23 = tf.cos(theta23)
s23 = tf.sin(theta23)

phase1=tf.complex(tf.cos(deltaCP),-tf.sin(deltaCP))
phase2=tf.complex(tf.cos(deltaCP),tf.sin(deltaCP))

U11 = tf.complex(c12*c13,0.0)
U12 = tf.complex(s12*c13,0.0)
U13 = tf.complex(s13,tf.constant(0.0))*phase1
U1 = tf.stack([U11,U12,U13])
U21 = tf.complex(-s12*c23,0.0) - tf.complex(c12*s23*s13,tf.constant(0.0))*phase2
U22 = tf.complex(c12*c23,0.0) - tf.complex(s12*s23*s13,tf.constant(0.0))*phase2
U23 = tf.complex(s23*c13,0.0)
U2 = tf.stack([U21,U22,U23])
U31 = tf.complex(s12*s23,0.0) - tf.complex(c12*c23*s13,tf.constant(0.0))*phase2
U32 = tf.complex(-c12*s23,0.0) - tf.complex(s12*c23*s13,tf.constant(0.0))*phase2
U33 = tf.complex(c23*c13,0.0)
U3 = tf.stack([U31,U32,U33])
U = tf.stack([U1,U2,U3])
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(U))

indices = [0, 1, 2]
depth = 3
id_mat=tf.cast(tf.one_hot(indices, depth),dtype=tf.complex64)
get_measured=[[.974,.225,.004],[.225,.973,.041],[.009,.04,.999]]
cost=tf.constant(0,dtype=tf.float32)
sess.run(tf.global_variables_initializer())
for i in range(3):
    a=id_mat[i]
    for j in range(3):
        b=id_mat[j]
        measured=tf.constant(get_measured[i][j],dtype=tf.complex64)
        cost = cost+tf.real(measured-tf.norm(tf.tensordot(a,tf.matmul(U,tf.expand_dims(b,0),adjoint_b=True,name="Ub"),axes=1,name="aUb")))**2

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
my_var_list=[theta12,theta13,theta23,deltaCP]
print(sess.run(my_var_list))
print(sess.run(cost))
opt_op = opt.minimize(cost, var_list=my_var_list)
opt_op.run(session=sess)
print(sess.run(my_var_list))
print(sess.run(cost))