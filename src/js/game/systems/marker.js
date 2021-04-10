import { GameSystemWithFilter } from "../game_system_with_filter";
import { StorageComponent } from "../components/storage";
import { DrawParameters } from "../../core/draw_parameters";
import { formatBigNumber, lerp } from "../../core/utils";
import { Loader } from "../../core/loader";
import { BOOL_TRUE_SINGLETON, BOOL_FALSE_SINGLETON } from "../items/boolean_item";
import { MapChunkView } from "../map_chunk_view";
import { MarkerComponent } from "../components/marker";
import { Rectangle } from "../../core/rectangle";
import { Entity } from "../entity";

export class MarkerSystem extends GameSystemWithFilter {
    constructor(root) {
        super(root, [MarkerComponent]);

        /**
         * Stores which uids were already drawn to avoid drawing entities twice
         * @type {Set<number>}
         */
        this.drawnUids = new Set();

        // using a simple set for this is not very efficient...
        // but I think it's probably fast enough, and javascript doesn't have geometric
        // data structures built-in iirc, so this will have to do
        /** @type {Set<{rect:Rectangle, ent:Entity}>} */
        this.markedAreas = new Set();

        this.root.signals.gameFrameStarted.add(this.clearDrawnUids, this);
    }

    clearDrawnUids() {
        this.drawnUids.clear();
    }

    update() {
        // I could set some event-based method up to remove and add them, todo
        this.markedAreas.clear();
        for (let i = 0; i < this.allEntities.length; i++) {
            let ent = this.allEntities[i];
            let staticComp = ent.components.StaticMapEntity;
            let markerComp = ent.components.Marker;
            let pos = staticComp.getTileSpaceBounds().getCenter().toWorldSpace();
            if (!markerComp.rect) {
                let xc = pos.x;
                let yc = pos.y;
                let x = xc - 16 + 32 * markerComp.x0;
                let y = yc - 16 + 32 * markerComp.y0;
                let w = xc - x + 16 + 32 * markerComp.x1;
                let h = yc - y + 16 + 32 * markerComp.y1;
                markerComp.rect = new Rectangle(x, y, w, h);
            }
            this.markedAreas.add({ rect: markerComp.rect, ent: ent });
        }
    }

    /**
     * @param {DrawParameters} parameters
     * @param {MapChunkView} chunk
     */
    drawChunk(parameters, chunk) {
        const contents = chunk.containedEntitiesByLayer.regular;
        for (let i = 0; i < contents.length; ++i) {
            const entity = contents[i];
            const markerComp = entity.components.Marker;
            if (!markerComp) {
                continue;
            }

            if (this.drawnUids.has(entity.uid)) {
                continue;
            }

            this.drawnUids.add(entity.uid);

            const staticComp = entity.components.StaticMapEntity;

            const context = parameters.context;
            const center = staticComp.getTileSpaceBounds().getCenter().toWorldSpace();
            context.font = "bold 16px GameFont";
            context.textAlign = "left";
            context.fillStyle = "#64666e";
            context.fillText(markerComp.title, center.x + 4, center.y - 10);
            context.globalAlpha = 1;
        }
    }
    /**
     * @param {DrawParameters} parameters
     * @param {MapChunkView} chunk
     */
    drawChunkHighlight(parameters, chunk) {
        this.markedAreas.forEach(pair => {
            let rect = pair.rect;
            let x0 = Math.max(rect.x, chunk.tileX * 32);
            let y0 = Math.max(rect.y, chunk.tileY * 32);
            let x1 = Math.min(rect.x + rect.w, (chunk.tileX + 16) * 32);
            let y1 = Math.min(rect.y + rect.h, (chunk.tileY + 16) * 32);
            if (x0 + 0.01 < x1 && y0 + 0.01 < y1) {
                const context = parameters.context;
                context.fillStyle = "#0000ff";
                context.globalAlpha = 0.05;
                context.fillRect(x0, y0, x1 - x0, y1 - y0);
                context.globalAlpha = 1;
            }
        });
    }
}
