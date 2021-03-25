import { RegularGameSpeed } from "./time/regular_game_speed";
import { gGameSpeedRegistry } from "../core/global_registries";
import { FastForwardGameSpeed } from "./time/fast_forward_game_speed";

export function initGameSpeedRegistry() {
    gGameSpeedRegistry.register(RegularGameSpeed);
    gGameSpeedRegistry.register(FastForwardGameSpeed);

    // Others are disabled for now
}
