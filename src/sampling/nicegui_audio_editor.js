export default {
    template: `
      <div :style="{ position: 'relative', aspectRatio: size ? size[0] / size[1] : undefined }">
        <img
          ref="img"
          :src="computed_src"
          :style="{ width: '100%', height: '100%', opacity: src ? 1 : 0 }"
          @load="onImageLoaded"
          v-on="onCrossEvents"
          v-on="onUserEvents"
          draggable="false"
        />
        <svg ref="svg" style="position:absolute;top:0;left:0;pointer-events:none" :viewBox="viewBox">
          <g :style="{ display: showCross ? 'block' : 'none' }">
            <line v-if="cross" :x1="x" y1="0" :x2="x" y2="100%" :stroke="cross === true ? 'black' : cross" />
            <line v-if="cross" x1="0" :y1="y" x2="100%" :y2="y" :stroke="cross === true ? 'black' : cross" />
            <slot name="cross" :x="x" :y="y"></slot>
          </g>
          <rect v-if="select_range" :x="sx" y="0" :width="sw" height="100%" fill="orange" fill-opacity="0.15" />
          <line v-if="select_range" :x1="sx" y1="0" :x2="sx" y2="100%" stroke="orange" />
          <line v-if="select_range" :x1="sx+sw" y1="0" :x2="sx+sw" y2="100%" stroke="orange" />
          <line :x1="tx" y1="0" :x2="tx" y2="100%" stroke="orange" stroke-width="2" />
          <g v-html="content"></g>
        </svg>
        <slot></slot>
      </div>
    `,
    data() {
      return {
        viewBox: "0 0 0 0",
        loaded_image_width: 0,
        loaded_image_height: 0,
        x: 100,
        y: 100,
        tx: 0, // time crosshair x position (in pixels)
        sx: 0, // selection start x position (in pixels)
        sw: 0, // selection width (in pixels)
        playing: false,
        time: 0,
        last_timestamp: 0,
        showCross: false,
        computed_src: undefined,
        waiting_source: undefined,
        loading: false,
      };
    },
    mounted() {
      setTimeout(() => this.compute_src(), 0); // NOTE: wait for window.path_prefix to be set in app.mounted()
      const handle_completion = () => {
        if (this.waiting_source) {
          this.computed_src = this.waiting_source;
          this.waiting_source = undefined;
        } else {
          this.loading = false;
        }
      };
      this.$refs.img.addEventListener("load", handle_completion);
      this.$refs.img.addEventListener("error", handle_completion);
      for (const event of [
        "pointermove",
        "pointerdown",
        "pointerup",
        "pointerover",
        "pointerout",
        "pointerenter",
        "pointerleave",
        "pointercancel",
      ]) {
        this.$refs.svg.addEventListener(event, (e) => this.onPointerEvent(event, e));
      }
    },
    updated() {
      this.compute_src();
    },
    methods: {
      compute_src() {
        const suffix = this.t ? (this.src.includes("?") ? "&" : "?") + "_nicegui_t=" + this.t : "";
        const new_src = (this.src.startsWith("/") ? window.path_prefix : "") + this.src + suffix;
        if (new_src == this.computed_src) {
          return;
        }
        if (this.loading) {
          this.waiting_source = new_src;
        } else {
          this.computed_src = new_src;
          this.loading = true;
        }
        if (!this.src && this.size) {
          this.viewBox = `0 0 ${this.size[0]} ${this.size[1]}`;
        }
      },
      updateCrossHair(e) {
        const width = this.src ? this.loaded_image_width : this.size ? this.size[0] : 1;
        const height = this.src ? this.loaded_image_height : this.size ? this.size[1] : 1;
        this.x = (e.offsetX * width) / e.target.clientWidth;
        this.y = (e.offsetY * height) / e.target.clientHeight;
      },
      onImageLoaded(e) {
        this.loaded_image_width = e.target.naturalWidth;
        this.loaded_image_height = e.target.naturalHeight;
        this.viewBox = `0 0 ${this.loaded_image_width} ${this.loaded_image_height}`;
        this.$emit("loaded", { width: this.loaded_image_width, height: this.loaded_image_height, source: e.target.src });
      },
      onMouseEvent(type, e) {
        //const imageWidth = this.src ? this.loaded_image_width : this.size ? this.size[0] : 1;
        //const imageHeight = this.src ? this.loaded_image_height : this.size ? this.size[1] : 1;
        this.$emit("mouse", {
          mouse_event_type: type,
          //image_x: (e.offsetX * imageWidth) / this.$refs.img.clientWidth,
          //image_y: (e.offsetY * imageHeight) / this.$refs.img.clientHeight,
          // modified to return normalized (0..1) coordinates
          image_x: e.offsetX / this.$refs.img.clientWidth,
          image_y: e.offsetY / this.$refs.img.clientHeight,
          button: e.button,
          buttons: e.buttons,
          altKey: e.altKey,
          ctrlKey: e.ctrlKey,
          metaKey: e.metaKey,
          shiftKey: e.shiftKey,
        });
      },
      onPointerEvent(type, e) {
        const imageWidth = this.src ? this.loaded_image_width : this.size ? this.size[0] : 1;
        const imageHeight = this.src ? this.loaded_image_height : this.size ? this.size[1] : 1;
        this.$emit(`svg:${type}`, {
          type: type,
          element_id: e.target.id,
          image_x: (e.offsetX * imageWidth) / this.$refs.svg.clientWidth,
          image_y: (e.offsetY * imageHeight) / this.$refs.svg.clientHeight,
        });
      },
      play(playing) {
        this.playing = playing;
        if (this.playing) {
            requestAnimationFrame((t) => this.update_time_cross(t));
        }
      },
      set_time(time) {
        this.time = time;
        this.last_timestamp = performance.now();
      },
      set_select_range(start, duration) {
        const pixels_per_second = this.loaded_image_width / this.duration;
        this.sx = start * pixels_per_second;
        this.sw = duration * pixels_per_second;
      },
      update_time_cross(timestamp) {
        if (this.playing) {
          const pixels_per_second = this.loaded_image_width / this.duration;
          const seconds_since_set_time = (timestamp - this.last_timestamp) / 1000;
          this.tx = (this.time + seconds_since_set_time) * pixels_per_second;
          requestAnimationFrame((t) => this.update_time_cross(t));
        }
      },
    },
    computed: {
      onCrossEvents() {
        if (!this.cross && !this.$slots.cross) return {};
        return {
          mouseenter: () => (this.showCross = true),
          mouseleave: () => (this.showCross = false),
          mousemove: (event) => this.updateCrossHair(event),
        };
      },
      onUserEvents() {
        const events = {};
        for (const type of this.events) {
          events[type] = (event) => this.onMouseEvent(type, event);
        }
        return events;
      },
    },
    props: {
      src: String,
      content: String,
      size: Object,
      events: Array,
      cross: Boolean,
      t: String,
      duration: Number,
      select_range: Boolean,
    },
  };
  