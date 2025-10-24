// AudioWorkletProcessor Audio Plugin
class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    this.frameSize = options?.processorOptions?.frameSize || 1024;
    this.buffer = [];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0][0];
    if (!input) return true;

    this.buffer.push(...input);

    while (this.buffer.length >= this.frameSize) {
      const frame = new Float32Array(this.buffer.slice(0, this.frameSize));
      this.buffer = this.buffer.slice(this.frameSize);

      this.port.postMessage({
        type: 'audioData',
        data: frame,
      }, [frame.buffer]);
    }
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);