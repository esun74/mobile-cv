import React, { useState, useEffect, useRef } from 'react'
import { StyleSheet, Text, View, TextInput, Image } from 'react-native'
import { Camera } from 'expo-camera'
import * as tf from '@tensorflow/tfjs'
import { bundleResourceIO, cameraWithTensors } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as blazeface from '@tensorflow-models/blazeface'
import jpeg from 'jpeg-js'

import modelJson from './nn_model/tfjs_model/model.json'
import modelWeights from './nn_model/tfjs_model/group1-shard1of1.bin'

export default function App() {

  const [hasPermission, setHasPermission] = useState(null)

	const [tfReady, setTfReady] = useState(false)
	const [sm, setSm] = useState(null)
	const [mm, setMm] = useState(null)
	const [cm, setCm] = useState(null)

	const TensorCamera = cameraWithTensors(Camera)
	let framecount = 0
	let framesperpred = 3
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

	const cnnp0 = useRef()
	const cnnp0_00 = useRef()
	const cnnp0_01 = useRef()
	const cnnp0_02 = useRef()
	const cnnp0_03 = useRef()
	const cnnp0_04 = useRef()
	const cnnp0_05 = useRef()
	const cnnp0_06 = useRef()
	const cnnp0_07 = useRef()
	const cnnp0_08 = useRef()
	const cnnp0_09 = useRef()
	const cnnp0_10 = useRef()
	const cnnp0_11 = useRef()
	const cnnp0_12 = useRef()
	const cnnp0_13 = useRef()
	const cnnp0_14 = useRef()

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
									marginLeft: pred[0].topLeft[0] * (640 / 196) + 15, width: (pred[0].bottomRight[0] - pred[0].topLeft[0]) * (640 / 196), 
									marginTop: pred[0].topLeft[1] * (640 / 196) + 45, height: (pred[0].bottomRight[1] - pred[0].topLeft[1]) * (640 / 196),
								}], text: (Math.round(pred[0].probability[0] * 10000) / 10000).toFixed(4)
							})
							bfp0_0.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[0][0] * (640 / 196) + 15, marginTop: pred[0].landmarks[0][1] * (640 / 196) + 45}]})
							bfp0_1.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[1][0] * (640 / 196) + 15, marginTop: pred[0].landmarks[1][1] * (640 / 196) + 45}]})
							bfp0_2.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[2][0] * (640 / 196) + 15, marginTop: pred[0].landmarks[2][1] * (640 / 196) + 45}]})
							bfp0_3.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[3][0] * (640 / 196) + 15, marginTop: pred[0].landmarks[3][1] * (640 / 196) + 45}]})
							bfp0_4.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[4][0] * (640 / 196) + 15, marginTop: pred[0].landmarks[4][1] * (640 / 196) + 45}]})
							bfp0_5.current.setNativeProps({style: [styles.rectangle, {marginLeft: pred[0].landmarks[5][0] * (640 / 196) + 15, marginTop: pred[0].landmarks[5][1] * (640 / 196) + 45}]})
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
						let greyscale = new Uint8Array(96 * 96)
						let greyscale_jpeg = new Uint8Array(96 * 96 * 4)

						for (let i = 0; i < 96; i++) {
							for (let j = 0; j < 96; j++) {
								greyscale[i * 96 + j] = Math.round(
									imageBuffer.values[((i + 48) * 108 + j + 6) * 3 + 0] * 0.2989 + 
									imageBuffer.values[((i + 48) * 108 + j + 6) * 3 + 1] * 0.5870 + 
									imageBuffer.values[((i + 48) * 108 + j + 6) * 3 + 2] * 0.1140
								)
								greyscale_jpeg[(i * 96 + j) * 4 + 0] = greyscale[i * 96 + j]
								greyscale_jpeg[(i * 96 + j) * 4 + 1] = greyscale[i * 96 + j]
								greyscale_jpeg[(i * 96 + j) * 4 + 2] = greyscale[i * 96 + j]
								greyscale_jpeg[(i * 96 + j) * 4 + 3] = 1 
							}
						}

						try {
							cnnp0.current.setNativeProps(
								{source:
									[{uri: 'data:image/jpeg;base64,' + jpeg.encode({data: greyscale_jpeg, width: 96, height: 96}).data.toString('base64')}]
								}
							)
						} catch (error) {
							console.warn(error)
						}

						let gs_tensor = tf.tensor4d(
							greyscale,
							[1, 96, 96, 1]
						)
						
						try {
							let pred = await cm.predict(gs_tensor, {batchSize: 1})
							pred = await pred.data()

							// top = 640
							// side = 360
							// dim = 192, 108
							// scaling = 640/192, 360/108
							// points taken = start at 48, 6 end at 144, 102 (padding: 48, 6)
							// scaled = 160, 20

							cnnp0_00.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[0] * (320 / 96) + 35, marginTop: pred[1] * (320 / 96) + 205}]})
							cnnp0_01.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[2] * (320 / 96) + 35, marginTop: pred[3] * (320 / 96) + 205}]})
							cnnp0_02.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[4] * (320 / 96) + 35, marginTop: pred[5] * (320 / 96) + 205}]})
							cnnp0_03.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[6] * (320 / 96) + 35, marginTop: pred[7] * (320 / 96) + 205}]})
							cnnp0_04.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[8] * (320 / 96) + 35, marginTop: pred[9] * (320 / 96) + 205}]})
							cnnp0_05.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[10] * (320 / 96) + 35, marginTop: pred[11] * (320 / 96) + 205}]})
							cnnp0_06.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[12] * (320 / 96) + 35, marginTop: pred[13] * (320 / 96) + 205}]})
							cnnp0_07.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[14] * (320 / 96) + 35, marginTop: pred[15] * (320 / 96) + 205}]})
							cnnp0_08.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[16] * (320 / 96) + 35, marginTop: pred[17] * (320 / 96) + 205}]})
							cnnp0_09.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[18] * (320 / 96) + 35, marginTop: pred[19] * (320 / 96) + 205}]})
							cnnp0_10.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[20] * (320 / 96) + 35, marginTop: pred[21] * (320 / 96) + 205}]})
							cnnp0_11.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[22] * (320 / 96) + 35, marginTop: pred[23] * (320 / 96) + 205}]})
							cnnp0_12.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[24] * (320 / 96) + 35, marginTop: pred[25] * (320 / 96) + 205}]})
							cnnp0_13.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[26] * (320 / 96) + 35, marginTop: pred[27] * (320 / 96) + 205}]})
							cnnp0_14.current.setNativeProps({style: [styles.redtangle, {marginLeft: pred[28] * (320 / 96) + 35, marginTop: pred[29] * (320 / 96) + 205}]})

						} catch (error) {
							console.error(error)
						}

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
						resizeHeight={192}
						resizeWidth={108}
						resizeDepth={0}
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


					<Image
						source={require('./random.jpg')}
						ref={cnnp0}
						style={{
							marginLeft: 35,
							marginTop: 205,
							width: 320,
							height: 320,
							position: 'absolute',
							zIndex: 50,
						}}
					/>


					<View ref={cnnp0_00}/>
					<View ref={cnnp0_01}/>
					<View ref={cnnp0_02}/>
					<View ref={cnnp0_03}/>
					<View ref={cnnp0_04}/>
					<View ref={cnnp0_05}/>
					<View ref={cnnp0_06}/>
					<View ref={cnnp0_07}/>
					<View ref={cnnp0_08}/>
					<View ref={cnnp0_09}/>
					<View ref={cnnp0_10}/>
					<View ref={cnnp0_11}/>
					<View ref={cnnp0_12}/>
					<View ref={cnnp0_13}/>
					<View ref={cnnp0_14}/>

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
	},
	redtangle: {
		height: 5,
		width: 5,
		backgroundColor: 'red',
		position: 'absolute',
		zIndex: 200,
	},
	box2: {
		position: 'absolute',
		borderWidth: 2,
		borderColor: 'red',
		color: 'red',
		zIndex: 200,
	},
})