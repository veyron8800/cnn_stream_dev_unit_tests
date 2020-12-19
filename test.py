from tensorflow import keras
import numpy as np
import os
import shutil
import hls4ml
import yaml

error_tolerance = 2**-5

activations = [
	keras.layers.Activation('sigmoid', name='sigmoid'),
	keras.layers.Activation('tanh', name='tanh'),
	keras.layers.Activation('hard_sigmoid', name='hard_sigmoid'),
	keras.layers.LeakyReLU(name='leaky_relu'),
	keras.layers.LeakyReLU(alpha=0.5, name='leaky_relu_2'),
	keras.layers.ThresholdedReLU(name='thresholeded_relu'),
	keras.layers.ThresholdedReLU(theta=0.5, name='thresholded_relu2'),
	keras.layers.Activation('softplus', name='softplus'),
	keras.layers.Activation('softsign', name='softsign'),
	keras.layers.ELU(name='elu'),
	keras.layers.ELU(alpha=3, name='elu2'),
	keras.layers.Activation('selu', name='selu'),
	keras.layers.PReLU(alpha_initializer='uniform', name='PReLU'),
	keras.layers.PReLU(alpha_initializer='normal', name='PReLU2')
]

if __name__ == '__main__':
	cfg = open('test.yml').read()

	for activation_layer in activations:
		m = keras.models.Sequential()
		m.add(keras.layers.Input((1000)))
		m.add(activation_layer)
		m.build()

		name = m.layers[0].name
		shutil.rmtree(name, ignore_errors=True)
		os.makedirs(name)
		os.chdir(name)

		m.save(name+'.h5')

		x = np.reshape(np.linspace(-5, 5, 1000), (1,1000))
		y = m.predict(x)

		os.makedirs('TestData')
		open('TestData/test_in.dat', 'w').write(' '.join(x.flatten().astype(str)))
		open('TestData/test_out.dat', 'w').write(' '.join(y.flatten().astype(str)))

		hls_model = hls4ml.converters.keras_to_hls(yaml.safe_load(cfg.format(name=name)))
		hls_model.compile()
		hls_model.build(csim=True, synth=False, cosim=False)

		csim_out = np.array(open(f'{name}/tb_data/csim_results.log').read().split(' ')[:-1]).astype(float)

		assert np.abs(csim_out - y).max() < error_tolerance, f'{name} failed csim validation'

		os.chdir('..')