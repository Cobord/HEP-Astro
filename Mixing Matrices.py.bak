import numpy as np
import tensorflow as tf

# CKM initialization
theta12Est = 13.04 * np.pi/180.0
theta23Est = .201* np.pi/180.0
theta13Est = 2.38* np.pi/180.0
deltaCPEst = 1.20
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