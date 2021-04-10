import { STOP_PROPAGATION } from "../../../core/signal";
import { Vector } from "../../../core/vector";
import { enumMouseButton } from "../../camera";
import { BaseHUDPart } from "../base_hud_part";
import { MarkerComponent } from "../../components/marker";
import { MarkerSystem } from "../../systems/marker";
import { MetaMarkerBuilding } from "../../buildings/marker";
import { DrawParameters } from "../../../core/draw_parameters";
import { Rectangle } from "../../../core/rectangle";

export class HUDResourceNodeEdit extends BaseHUDPart {
    initialize() {
        this.root.camera.downPreHandler.add(this.downPreHandler, this);
        this.root.camera.upPostHandler.add(this.upPostHandler, this);
        this.root.camera.movePreHandler.add(this.movePreHandler, this);
        this.dragging = null;
    }

    /**
     * @param {Vector} pos
     * @param {enumMouseButton} button
     */
    downPreHandler(pos, button) {
        if (button === enumMouseButton.left) {
            const tile = this.root.camera.screenToWorld(pos).toTileSpace();
            if (!this.root.map.getLayerContentXY(tile.x, tile.y, "regular")) {
                const contents = this.root.map.getLowerLayerContentXY(tile.x, tile.y);
                if (contents) {
                    this.dragging = { x: tile.x, y: tile.y };
                    return STOP_PROPAGATION;
                }
            }
        }
    }
    upPostHandler(pos, button) {
        if (this.dragging) {
            if (this.isValidDropPos(pos)) {
                const target = this.root.camera.screenToWorld(pos).toTileSpace();
                const source = this.root.map.getLowerLayerContentXY(this.dragging.x, this.dragging.y);
                this.root.map.setLowerLayerContentXY(target.x, target.y, source);
                this.root.map.setLowerLayerContentXY(this.dragging.x, this.dragging.y, null);
            }
            this.dragging = null;
            return STOP_PROPAGATION;
        }
    }
    movePreHandler(vec) {
        this.cursor = vec;
    }

    isValidDropPos(pos) {
        const target = this.root.camera.screenToWorld(pos).toTileSpace();
        return (
            !this.root.map.getLowerLayerContentXY(target.x, target.y) &&
            !this.root.map.getLayerContentXY(target.x, target.y, this.root.currentLayer)
        );
    }
    /**
     * Should draw the hud
     * @param {DrawParameters} parameters
     */
    draw(parameters) {
        if (this.dragging) {
            const source = this.root.map.getLowerLayerContentXY(this.dragging.x, this.dragging.y);
            const pos = this.root.camera.screenToWorld(this.cursor);
            if (this.isValidDropPos(this.cursor)) {
                const tile = pos.snapWorldToTile();
                source.drawItemCenteredClipped(tile.x + 16, tile.y + 16, parameters);
            }
            source.drawItemCenteredClipped(pos.x, pos.y, parameters);
        }
    }
}
