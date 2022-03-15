import React, { useState, useEffect, useRef } from 'react'
import { StyleSheet, Text, View, TextInput } from 'react-native'
import { Camera } from 'expo-camera'
import * as tf from '@tensorflow/tfjs'
import { bundleResourceIO, cameraWithTensors } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as blazeface from '@tensorflow-models/blazeface'

import modelJson from './nn_model/tfjs_model/model.json'
import modelWeights from './nn_model/tfjs_model/group1-shard1of1.bin'

export default function App() {

  const [hasPermission, setHasPermission] = useState(null);

	const [tfReady, setTfReady] = useState(false)
	const [sm, setSm] = useState(null)
	const [mm, setMm] = useState(null)
	const [cm, setCm] = useState(null)

	const TensorCamera = cameraWithTensors(Camera)
	let framecount = 0
	let framesperpred = 10
	let frameid = 0

	const sm_predictions = useRef()
	const sm_probabilities = useRef()

	const bfp0 = useRef()
	const bfp0_0 = useRef()
	const bfp0_1 = useRef()
	const bfp0_2 = useRef()
	const bfp0_3 = useRef()
	const bfp0_4 = useRef()
	const bfp0_5 = useRef()


	// Request permissions
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync()
      setHasPermission(status === 'granted')
    })()
  }, [])

	// Load TensorFlow + models
	useEffect( async () => {
		tf.ready().then(async () => {
			setTfReady(true)
			mobilenet.load().then(setSm)
			blazeface.load().then(setMm)
			tf.loadLayersModel(
				bundleResourceIO(
					modelJson, 
					modelWeights
				)
			).then(setCm)
		})
	}, [])

	function handleCameraStream(image) {
		if (image) {
			const loop = async () => {

				let imageTensor = image.next().value

				if (imageTensor) {
					if (framecount === 0 && sm) {

						let pred = await sm.classify(imageTensor).catch(console.log)
						sm_predictions.current.setNativeProps({
							text: 'MobileNet Predictions \n ' + pred.map(e => e.className.replace(/(.{35})..+/, "$1…")).join('\n ')
						})
						sm_probabilities.current.setNativeProps({
							text: 'Probabilities\n' + pred.map(e => (Math.round(e.probability * 10000) / 10000).toFixed(4)).join('\n')
						})

					} else if (framecount === ((framesperpred / 3) >> 0) && mm) {

						let pred = await mm.estimateFaces(imageTensor).catch(console.log)
						
						if (pred[0]) {
							bfp0.current.setNativeProps({
								style: [styles.box, {
									marginLeft: pred[0].topLeft[0] * 2 + 15, width: (pred[0].bottomRight[0] - pred[0].topLeft[0]) * 2, 
									marginTop: pred[0].topLeft[1] * 2 + 45, height: (pred[0].bottomRight[1] - pred[0].topLeft[1]) * 2,
								}], text: (Math.round(pred[0].probability[0] * 10000) / 10000).toFixed(4)
							})
							bfp0_0.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[0][0] * 2 + 15, marginTop: pred[0].landmarks[0][1] * 2 + 45}]})
							bfp0_1.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[1][0] * 2 + 15, marginTop: pred[0].landmarks[1][1] * 2 + 45}]})
							bfp0_2.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[2][0] * 2 + 15, marginTop: pred[0].landmarks[2][1] * 2 + 45}]})
							bfp0_3.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[3][0] * 2 + 15, marginTop: pred[0].landmarks[3][1] * 2 + 45}]})
							bfp0_4.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[4][0] * 2 + 15, marginTop: pred[0].landmarks[4][1] * 2 + 45}]})
							bfp0_5.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[5][0] * 2 + 15, marginTop: pred[0].landmarks[5][1] * 2 + 45}]})
						} else {
							bfp0.current.setNativeProps({style: [styles.box, {marginLeft: 0, marginTop: 0, width: 0, height: 0}], text: ''})
							bfp0_0.current.setNativeProps({style: [styles.rectangle, {marginLeft: 0, marginTop: 0}]})
							bfp0_1.current.setNativeProps({style: [styles.rectangle, {marginLeft: 0, marginTop: 0}]})
							bfp0_2.current.setNativeProps({style: [styles.rectangle, {marginLeft: 0, marginTop: 0}]})
							bfp0_3.current.setNativeProps({style: [styles.rectangle, {marginLeft: 0, marginTop: 0}]})
							bfp0_4.current.setNativeProps({style: [styles.rectangle, {marginLeft: 0, marginTop: 0}]})
							bfp0_5.current.setNativeProps({style: [styles.rectangle, {marginLeft: 0, marginTop: 0}]})
						}

					} else if (framecount === (((framesperpred * 2 / 3) >> 0)) && cm) {

						let imageBuffer = await imageTensor.buffer()
						let greyscale = new Uint8Array(imageBuffer.shape[0] * imageBuffer.shape[1])

						for (let i = 0; i < imageBuffer.shape[0] * imageBuffer.shape[1]; i++) {
							greyscale[i] = 
								imageBuffer.values[3 * i + 0] * 0.2989 + 
								imageBuffer.values[3 * i + 1] * 0.5870 + 
								imageBuffer.values[3 * i + 2] * 0.1140
						}

						let gs_tensor = tf.tensor4d(
							greyscale,
							[1, imageBuffer.shape[0], imageBuffer.shape[1], 1]
						)
						
						// console.warn(gs_tensor)
						// console.warn(cm.predict(gs_tensor))
						// cm.predict(gs_tensor).catch(console.error)
						// let predictions = await cm.predict(gs_tensor).catch(console.error)
						// predictions.data().then(console.warn)

						tf.dispose(gs_tensor)
					}
				}


				tf.dispose(imageTensor)
				framecount = (framecount + 1) % framesperpred

				frameid = requestAnimationFrame(loop)
			}
			loop()
		}
	}

	useEffect(() => {
		return () => cancelAnimationFrame(frameid)
	}, [ frameid ])

	// Deal with situations surrounding no / awaiting permissions
  if (hasPermission === null) {
    return <View />
  }

  if (hasPermission === false) {
    return <View style={styles.middletext}>
			<Text>No access to camera</Text>
		</View>
  }

	// GUI
  return (
    <>
			{tfReady && sm && mm && cm ? 
				<>
					<TensorCamera
						// iPhone 16:9 ratio, otherwise it will strech
						style={styles.camera}
						type={Camera.Constants.Type.front}
						cameraTextureHeight={1920}
						cameraTextureWidth={1080}
						resizeHeight={320}
						resizeWidth={180}
						resizeDepth={3}
						onReady={handleCameraStream}
						autorender={true}
					/>

					<TextInput 
						multiline
						editable={false} 
						style={styles.bottomleft}
						ref={sm_predictions} 
						text='mobilenet'
					/>
					<TextInput 
						multiline
						editable={false} 
						style={styles.bottomright}
						ref={sm_probabilities} 
						text='probabilities'
					/>

					<TextInput
						multiline
						editable={false}
						ref={bfp0}
					/>
					<View ref={bfp0_0}/>
					<View ref={bfp0_1}/>
					<View ref={bfp0_2}/>
					<View ref={bfp0_3}/>
					<View ref={bfp0_4}/>
					<View ref={bfp0_5}/>
				</>
			:
			<View style={styles.middletext}>
				<Text>
					Loading...
				</Text>
			</View>
			}
    </>
  )
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
	camera: {
    position: "absolute",
    left: 15,
    top: 45,
    width: 360,
    height: 640,
    zIndex: -100,
	},
	bottomleft: {
		position: "absolute",
		left: 15,
		top: 700,
		width: 300,
		height: 100,
		zIndex: 0,
		color: 'black',
	},
	bottomright: {
		position: "absolute",
		right: 15,
		top: 700,
		width: 100,
		height: 100,
		zIndex: 0,
		color: 'black',
		textAlign: 'right',
	},
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
		padding: 50,
  },
  text: {
    fontSize: 18,
    color: 'black',
  },
	middletext: {
		flex: 1,
		alignItems: 'center',
		justifyContent: 'center',
	},
	rectangle: {
		height: 5,
		width: 5,
		backgroundColor: 'mediumseagreen',
		position: 'absolute',
		zIndex: 100,
	},
	box: {
		position: 'absolute',
		borderWidth: 2,
		borderColor: 'mediumseagreen',
		color: 'mediumseagreen',
		zIndex: 100,
	}
})